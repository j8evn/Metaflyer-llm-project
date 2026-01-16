import os
import json
import glob
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# 학습 데이터를 Validation 폴더로 변경
TRAIN_BASE_DIR = "/dataset/cep/37.한국적_영상_이해_데이터/3.개방데이터/1.데이터/Validation"
IMG_DIR = os.path.join(TRAIN_BASE_DIR, "01.원천데이터")
LABEL_DIR = os.path.join(TRAIN_BASE_DIR, "02.라벨링데이터")

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
OUTPUT_DIR = "./qwen3_vl_lora_results"

# 1. 데이터 준비 (Dataset)
class Qwen3Dataset(torch.utils.data.Dataset):
    def __init__(self, label_files, processor):
        self.label_files = label_files
        self.processor = processor

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_path = self.label_files[idx]
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_base = os.path.basename(label_path).replace('.json', '')
        img_path = os.path.join(IMG_DIR, file_base + '.png')
        
        if not os.path.exists(img_path):
            img_path = None # 에러 처리 필요

        # 키워드 추출
        gt_keywords = []
        for i in range(1, 4):
            cat = data['image'].get(f'image_category_{i}')
            if cat and cat.strip():
                gt_keywords.append(cat.strip())
        
        target_text = ",".join(gt_keywords)

        # 메시지 포맷
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": "이 이미지에서 핵심 키워드를 추출해줘."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        ]
        return messages

def data_collator(features, processor):
    # features: list of messages
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in features]
    image_inputs, video_inputs = process_vision_info(features)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # [개선] 레이블 마스킹: Assistant 답변 부분에만 Loss 계산
    # <|im_start|>assistant\n 부분을 찾아 그 이후만 학습하도록 설정
    labels = inputs["input_ids"].clone()
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>assistant")
    
    for i in range(labels.shape[0]):
        # assistant 시작 위치 찾기
        input_ids = labels[i].tolist()
        try:
            # 보통 <|im_start|>assistant\n 다음부터 대답이 시작됨
            # 실제로는 템플릿에 따라 다르므로 간단히 assistant 토큰 위치 이후를 대상 처리
            sep_idx = input_ids.index(assistant_token_id) + 1
            # 질문 부분(0 ~ sep_idx)은 -100으로 마스킹하여 Loss 계산 제외
            labels[i, :sep_idx+1] = -100 
        except ValueError:
            pass # 못 찾으면 전체 학습 (안전장치)
            
    inputs["labels"] = labels
    return inputs

# 2. 실행 루틴
def train():
    print("\n[1/4] 모델 로드 및 LoRA 설정...")
    torch.cuda.empty_cache() # 기존 캐시 정리
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 각 GPU가 65GB 이상 사용하지 않도록 제한 (나머지는 학습 공간으로 활용)
    # CUDA_VISIBLE_DEVICES=1,2,3 인 경우 0, 1, 2로 인식됨
    max_memory = {0: "65GiB", 1: "65GiB", 2: "65GiB"}

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("\n[2/4] 데이터셋 준비...")
    all_labels = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))
    # Validation 전체 데이터(약 4,200건) 학습
    train_labels = all_labels 
    train_dataset = Qwen3Dataset(train_labels, processor)

    print(f"학습 데이터: {len(train_dataset)}개")

    print("\n[3/4] Trainer 설정 및 학습 시작...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,             # 안정적인 학습을 위해 약간 낮춤
        num_train_epochs=5,              # 확실한 학습을 위해 에폭 증가
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: data_collator(x, processor),
    )

    trainer.train()

    print("\n[4/4] 모델 저장 중...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    print("완료!")

if __name__ == "__main__":
    train()
