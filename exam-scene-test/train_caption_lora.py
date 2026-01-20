"""
Qwen3-VL-30B-A3B-Instruct 이미지 캡셔닝 LoRA 학습 스크립트

사용법:
    # 전체 학습
    python train_caption_lora.py

    # 테스트 학습 (1000건, 1 epoch)
    python train_caption_lora.py --test

    # 커스텀 샘플 수
    python train_caption_lora.py --max_samples 5000 --epochs 2

데이터 구조:
    /dataset/cep/training/      # 이미지(.png)와 JSON(.json)이 같이 있음
    /dataset/cep/validation/data/
"""

import os
import json
import glob
import random
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

# ============ 설정 ============
# Hugging Face 캐시 경로 (모델이 다운로드된 위치)
HF_CACHE_DIR = "/dataset/cep/cache/huggingface/hub"

TRAIN_DATA_DIR = "/dataset/cep/training"       # 이미지(.png)와 JSON(.json)이 같이 있는 디렉토리
VAL_DATA_DIR = "/dataset/cep/validation/data"  # 검증 데이터 디렉토리

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
OUTPUT_DIR = "./output_caption"

# 캡셔닝 프롬프트 (다양성을 위해 여러 개 사용)
CAPTION_PROMPTS = [
    "이 이미지를 자세히 설명해주세요.",
    "이 사진에서 무슨 일이 일어나고 있나요?",
    "이 이미지의 내용을 한국어로 설명해주세요.",
    "이 장면을 묘사해주세요.",
    "이 이미지에 보이는 것을 설명해주세요.",
]


class CaptionDataset(torch.utils.data.Dataset):
    """이미지 캡셔닝용 데이터셋 (개별 JSON 파일 기반)"""

    def __init__(self, data_dir: str, processor, caption_lang: str = "ko", max_samples: int = None):
        self.data_dir = data_dir  # 이미지와 JSON이 같은 디렉토리에 있음
        self.processor = processor
        self.caption_lang = caption_lang

        # 라벨 파일 수집 (JSON 파일 기준)
        self.label_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

        # 샘플 수 제한 (테스트용)
        if max_samples and max_samples < len(self.label_files):
            self.label_files = self.label_files[:max_samples]

        print(f"Found {len(self.label_files)} label files in {data_dir}")

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_path = self.label_files[idx]

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            return self._get_dummy_item()

        # 이미지 경로 찾기 (같은 디렉토리에서 같은 이름의 .png 파일)
        file_base = os.path.basename(label_path).replace('.json', '')
        img_path = os.path.join(self.data_dir, file_base + '.png')

        if not os.path.exists(img_path):
            # jpg 시도
            img_path = os.path.join(self.data_dir, file_base + '.jpg')
            if not os.path.exists(img_path):
                print(f"Image not found: {file_base}")
                return self._get_dummy_item()

        # 캡션 추출 (caption_ko_1~5 중 랜덤 선택)
        caption = self._extract_caption(data)
        if not caption:
            print(f"No caption found in {label_path}")
            return self._get_dummy_item()

        # 프롬프트 랜덤 선택
        prompt = random.choice(CAPTION_PROMPTS)

        # Qwen3-VL 메시지 형식
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            }
        ]

        return messages

    def _extract_caption(self, data: dict) -> str:
        """JSON에서 캡션 추출 (caption_ko_1~5 중 랜덤 선택)"""
        context = data.get('context', {})
        prefix = f"caption_{self.caption_lang}_"

        captions = []
        for i in range(1, 6):
            key = f"{prefix}{i}"
            cap = context.get(key, "")
            if cap and cap.strip():
                captions.append(cap.strip())

        if not captions:
            return ""

        return random.choice(captions)

    def _get_dummy_item(self):
        """에러 발생 시 더미 아이템 반환"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "dummy"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "dummy"}],
            }
        ]


def data_collator(features, processor):
    """배치 데이터 처리 및 레이블 마스킹"""
    # 더미 아이템 필터링
    valid_features = [f for f in features if f[0]['content'][0].get('type') != 'text' or f[0]['content'][0].get('text') != 'dummy']

    if not valid_features:
        return None

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in valid_features]
    image_inputs, video_inputs = process_vision_info(valid_features)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 레이블 마스킹: Assistant 답변 부분에만 Loss 계산
    labels = inputs["input_ids"].clone()

    # <|im_start|>assistant 토큰 ID 찾기
    try:
        assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    except:
        assistant_token_id = None

    if assistant_token_id is not None:
        for i in range(labels.shape[0]):
            input_ids = labels[i].tolist()
            try:
                # assistant 시작 위치 찾기
                sep_idx = input_ids.index(assistant_token_id)
                # 다음 토큰이 'assistant'인지 확인하고 그 이후부터 학습
                labels[i, :sep_idx+2] = -100  # 질문 부분 마스킹
            except ValueError:
                pass

    # 패딩 토큰 마스킹
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = labels
    return inputs


def train(args):
    print("=" * 60)
    print("Qwen3-VL 이미지 캡셔닝 LoRA 학습")
    if args.test:
        print("[테스트 모드] 1000건, 1 epoch")
    print("=" * 60)

    # 1. 모델 로드
    print("\n[1/4] 모델 로드 중...")
    torch.cuda.empty_cache()

    # 4-bit 양자화 설정 (메모리 최적화)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # GPU 메모리 제한 (필요시 조정)
    max_memory = {i: "70GiB" for i in range(torch.cuda.device_count())}

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=HF_CACHE_DIR,
    )

    # prepare_model_for_kbit_training 대신 직접 설정 (메모리 절약)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # 모든 파라미터를 학습 불가로 설정 (LoRA만 학습)
    for param in model.parameters():
        param.requires_grad = False

    # 2. LoRA 설정
    print("\n[2/4] LoRA 설정 적용 중...")
    if args.test:
        # 테스트 모드: 더 낮은 가중치 (빠른 테스트)
        lora_r = 8
        lora_alpha = 16
    else:
        # 전체 학습: 더 높은 가중치
        lora_r = 16
        lora_alpha = 32

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"LoRA 설정: r={lora_r}, alpha={lora_alpha}")

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE_DIR)

    # 3. 데이터셋 준비
    print("\n[3/4] 데이터셋 준비 중...")
    train_dataset = CaptionDataset(TRAIN_DATA_DIR, processor, max_samples=args.max_samples)
    print(f"학습 데이터: {len(train_dataset)}건")

    # 테스트 모드에서는 검증 생략 (빠른 테스트)
    if args.test:
        val_dataset = None
        print("검증 데이터: 생략 (테스트 모드)")
    else:
        val_dataset = CaptionDataset(VAL_DATA_DIR, processor, max_samples=args.max_val_samples)
        print(f"검증 데이터: {len(val_dataset)}건")

    # 4. 학습 설정
    print("\n[4/4] 학습 시작...")
    output_dir = args.output_dir or (OUTPUT_DIR + "_test" if args.test else OUTPUT_DIR)

    # 테스트 모드에서는 더 자주 로깅/저장, 검증 생략
    logging_steps = 10 if args.test else 50
    save_steps = 100 if args.test else 500

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # 배치 설정
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16

        # 학습률
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        # Epoch
        num_train_epochs=args.epochs,

        # 최적화
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",

        # 로깅 및 저장
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,

        # 검증 (테스트 모드에서는 생략)
        eval_strategy="no" if args.test else "steps",
        eval_steps=500,

        # 기타
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        # 리포팅
        report_to="none",  # wandb 사용 시 "wandb"로 변경
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda x: data_collator(x, processor),
    )

    # 학습 실행
    trainer.train()

    # 모델 저장
    print("\n모델 저장 중...")
    save_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"어댑터 저장 위치: {save_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 이미지 캡셔닝 LoRA 학습")

    parser.add_argument("--test", action="store_true",
                        help="테스트 모드 (1000건, 1 epoch)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="학습 데이터 최대 샘플 수 (기본: 전체)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="검증 데이터 최대 샘플 수 (기본: 전체)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="학습 epoch 수 (기본: 3)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 디렉토리 (기본: ./output_caption)")

    args = parser.parse_args()

    # 테스트 모드 기본값 설정
    if args.test:
        if args.max_samples is None:
            args.max_samples = 1000
        if args.max_val_samples is None:
            args.max_val_samples = 100
        if args.epochs == 3:  # 기본값이면 1로 변경
            args.epochs = 1

    train(args)


if __name__ == "__main__":
    main()

