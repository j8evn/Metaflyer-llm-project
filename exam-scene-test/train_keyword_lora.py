"""
Qwen3-VL-30B-A3B-Instruct 이미지 키워드 분류 LoRA 학습 스크립트

사용법:
    # 전체 학습
    CUDA_VISIBLE_DEVICES=2 python train_category_lora.py

    # 테스트 학습 (100건, 1 epoch)
    CUDA_VISIBLE_DEVICES=2 python train_category_lora.py --test

    # 커스텀 설정
    CUDA_VISIBLE_DEVICES=2 python train_category_lora.py --max_samples 500 --epochs 2

데이터:
    /dataset/cep/validation/data/ - 이미지(.png)와 JSON(.json)
    JSON에서 caption_ko_1~5 캡션 명사 추출 (analytics.py와 동일한 로직)
"""

import os
import json
import glob
import argparse
import torch
from collections import Counter
from tqdm import tqdm
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
HF_CACHE_DIR = "/dataset/cep/cache/huggingface/hub"

# 데이터 경로
DATA_DIR = "/dataset/cep/validation/data"  # 이미지(.png)와 JSON(.json)이 같이 있음

# 모델 설정 (우선순위: 병합 모델 > Base 모델)
MERGED_MODEL_PATH = "/dataset/cep/vl/merged_model"  # 캡셔닝 학습된 모델
BASE_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
OUTPUT_DIR = "/dataset/cep/vl/output_category"

# 키워드 설정
MAX_KEYWORDS = 1000  # 상위 1000개 키워드 사용
TOP_KEYWORDS_TO_ADD = 3  # 각 이미지에 추가할 상위 키워드 수

# 이미지 해상도 설정 (학습 속도 vs 정확도 트레이드오프)
# 학습: 낮은 해상도로 빠르게, 추론: 높은 해상도로 정확하게
MAX_PIXELS = 65536   # 256*256 (학습 속도 우선)
MIN_PIXELS = 28 * 28  # 최소 해상도

# 키워드 분류 프롬프트 템플릿 (analytics.py와 동일한 형식)
KEYWORD_PROMPT_TEMPLATE = """다음 목록에서 이미지에 보이는 키워드를 1~6개 선택하세요.

[목록]
{keywords}

[응답 예시]
{examples}

위 예시처럼 쉼표로 구분하여 답하세요. 목록에 없는 단어는 사용하지 마세요."""


# 제외할 단어 (비시각적, 추상적, Okt 오분류) - analytics.py와 동일
STOPWORDS = {
    # 일반 대명사/의존명사
    '것', '수', '때', '곳', '등', '씨', '중', '내', '위', '뒤', '앞', '옆',
    '이', '그', '저', '무엇', '어디', '누구', '언제', '어떻게',
    # 추상적 단어
    '모습', '형태', '상태', '느낌', '배경', '전경', '부분', '전체', '일부',
    '장면', '상황', '모양', '분위기', '광경', '풍경',
    # 너무 일반적인 단어
    '사람', '남자', '여자', '아이', '사물', '물체', '물건',
    # 방향/위치 (시각적이지 않음)
    '오른쪽', '왼쪽', '가운데', '중앙', '주변', '근처', '멀리', '가까이',
    '위로', '아래로', '옆으로', '안쪽', '바깥쪽',
    # Okt 오분류 (동사/형용사를 명사로 잘못 추출)
    '보이', '있어', '하고', '되어', '위해', '통해', '따라', '대해',
    '여러', '다양', '많은', '적은', '큰', '작은', '높은', '낮은',
    '새로운', '오래된', '특별', '일반', '전체적',
    # 수량/정도 표현
    '명의', '개의', '마리', '대의', '채의', '장의',
    '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열',
    # 기타 비시각적
    '취하', '찾기', '휴식', '협동', '대비', '무장', '매혹', '깊이', '구경',
    # 로그 분석을 통해 추가된 비시각적/추상적 단어
    '뒤쪽', '무언가', '약간', '긴장감', '고요한', '사이', '사이사이',
    '형성', '보기', '구조', '질서', '현대', '실내', '반사', '전시',
    # 추가 발견된 노이즈 (로그 기반)
    '아래', '공간', '전통', '착용', '다른', '요리', '작업', '자연', '남성은', '환경',
    '설치', '장식', '가득', '준비', '보아', '집중', '배경', '물이', '잡고', '사용',
    '정리', '사진', '야외', '식사', '모두', '대화', '강조', '거나', '두운', '색상', '무늬',
    '내부',
    # 3차 필터링
    '한국', '활동', '날씨', '디자인', '대조', '진행', '그녀', '서서', '주위', '조리', '배치',
    # 4차 필터링 (상위 200개 기반)
    '매력', '표정', '맑은', '위치', '매우', '장소', '행사', '중인', '더욱', '표시',
    '너머', '양쪽', '지역', '구성', '보고', '촬영', '복장', '시간', '살짝', '보이지',
    '연결', '바탕', '참여', '자리', '건축', '포함', '요소', '타고', '설명', '역사',
    '종류', '직사각형', '질감', '위생', '암시', '대형', '각각', '이야기', '대가', '인물',
    '색깔', '가능성', '정돈', '특징', '재질', '몇몇', '배열', '테두리', '한쪽', '앞쪽',
    '정면', '무성', '조물',
    # 5차 필터링
    '바닥', '천장', '용기', '여러가지', '다양한', '여러', '검은색', '흰색', '남성', '여성',
}


def extract_keywords_from_data(data_dir: str) -> list:
    """데이터에서 키워드 목록 추출 (캡션 명사만 사용) - analytics.py와 동일"""
    from konlpy.tag import Okt
    okt = Okt()

    label_files = glob.glob(os.path.join(data_dir, "*.json"))
    all_keywords = Counter()

    for label_path in tqdm(label_files, desc="키워드 추출"):
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        # caption_ko_1~5에서 명사 추출
        context = data.get("context", {})
        for i in range(1, 6):
            caption = context.get(f"caption_ko_{i}", "")
            if caption:
                nouns = okt.nouns(caption)
                for noun in nouns:
                    if len(noun) >= 2 and noun not in STOPWORDS:
                        all_keywords[noun] += 1

    # 빈도순 상위 MAX_KEYWORDS개
    result = []
    seen = set()
    for kw, _ in all_keywords.most_common():
        normalized = kw.strip().replace("'", "").replace('"', "").replace(" ", "")
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
            if len(result) >= MAX_KEYWORDS:
                break

    print(f"  - 총 키워드: {len(result)}개")
    return result


class CategoryDataset(torch.utils.data.Dataset):
    """이미지 키워드 분류용 데이터셋 (analytics.py 로직과 동일)"""

    def __init__(self, data_dir: str, processor, keyword_list: list, max_samples: int = None):
        from konlpy.tag import Okt
        self.okt = Okt()

        self.data_dir = data_dir
        self.processor = processor
        self.keyword_list = keyword_list  # 전체 키워드 목록 (빈도순)
        self.keyword_set = set(keyword_list)  # 빠른 검색용

        # 라벨 파일 수집
        self.label_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

        # 샘플 수 제한
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

        # 이미지 경로 찾기
        file_base = os.path.basename(label_path).replace('.json', '')
        img_path = os.path.join(self.data_dir, file_base + '.png')

        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, file_base + '.jpg')
            if not os.path.exists(img_path):
                print(f"Image not found: {file_base}")
                return self._get_dummy_item()

        # 해당 이미지의 키워드 목록 추출 (캡션에서)
        image_keywords = self._extract_image_keywords(data)

        # 전체 상위 키워드 추가 (중복 제외) - analytics.py와 동일
        for top_kw in self.keyword_list[:TOP_KEYWORDS_TO_ADD]:
            if top_kw not in image_keywords:
                image_keywords.append(top_kw)

        if not image_keywords:
            return self._get_dummy_item()

        # GT 키워드 추출 (학습 정답으로 사용)
        gt_keywords = self._extract_gt_keywords(data, set(image_keywords))
        if not gt_keywords:
            return self._get_dummy_item()

        # 키워드 문자열 (쉼표로 구분, 최대 6개)
        keyword_answer = ", ".join(gt_keywords[:6])

        # 프롬프트 구성 (analytics.py와 동일한 형식)
        example_keywords = image_keywords[:3]
        example_str = ", ".join(example_keywords)
        prompt = KEYWORD_PROMPT_TEMPLATE.format(
            keywords=", ".join(image_keywords),
            examples=example_str
        )

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
                "content": [{"type": "text", "text": keyword_answer}],
            }
        ]

        return messages

    def _extract_image_keywords(self, data: dict) -> list:
        """JSON 캡션에서 해당 이미지의 키워드 목록 추출 (analytics.py와 동일)"""
        image_keywords = []

        # caption_ko_1~5에서 명사 추출
        context = data.get('context', {})
        for i in range(1, 6):
            caption = context.get(f'caption_ko_{i}', '')
            if caption:
                nouns = self.okt.nouns(caption)
                for noun in nouns:
                    # keyword_set에 있는 것만 (전체 키워드 목록에 있는 것)
                    if len(noun) >= 2 and noun in self.keyword_set and noun not in image_keywords:
                        image_keywords.append(noun)

        return image_keywords

    def _extract_gt_keywords(self, data: dict, image_keyword_set: set) -> list:
        """JSON 캡션에서 GT 키워드 추출 (학습 정답용)"""
        gt_keywords = []

        # caption_ko_1~5에서 명사 추출
        context = data.get('context', {})
        for i in range(1, 6):
            caption = context.get(f'caption_ko_{i}', '')
            if caption:
                nouns = self.okt.nouns(caption)
                for noun in nouns:
                    # image_keyword_set에 있는 것만 (해당 이미지의 키워드 목록에 있는 것)
                    if noun in image_keyword_set and noun not in gt_keywords:
                        gt_keywords.append(noun)

        return gt_keywords

    def _get_dummy_item(self):
        """에러 발생 시 더미 아이템"""
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": "dummy"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "dummy"}],
            }
        ]


def data_collator(features, processor):
    """배치 데이터 처리 및 레이블 마스킹"""
    # 더미 필터링
    valid_features = [
        f for f in features
        if f[0]['content'][0].get('type') != 'text' or f[0]['content'][0].get('text') != 'dummy'
    ]

    if not valid_features:
        return None

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in valid_features
    ]
    image_inputs, video_inputs = process_vision_info(valid_features)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 레이블 마스킹: Assistant 답변만 학습
    labels = inputs["input_ids"].clone()

    try:
        assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    except:
        assistant_token_id = None

    if assistant_token_id is not None:
        for i in range(labels.shape[0]):
            input_ids = labels[i].tolist()
            try:
                sep_idx = input_ids.index(assistant_token_id)
                labels[i, :sep_idx+2] = -100  # 질문 부분 마스킹
            except ValueError:
                pass

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = labels
    return inputs


def train(args):
    print("=" * 60)
    print("Qwen3-VL 이미지 키워드 분류 LoRA 학습")
    if args.test:
        print("[테스트 모드]")
    print("=" * 60)

    # 키워드 목록 추출 (캡션 명사만 사용) - analytics.py와 동일
    print("\n[0/4] 키워드 목록 추출 중 (캡션 명사)...")
    all_keywords = extract_keywords_from_data(DATA_DIR)
    keyword_list = all_keywords[:MAX_KEYWORDS]
    print(f"총 {len(all_keywords)}개 키워드 중 상위 {len(keyword_list)}개 사용")

    # 1. 모델 로드
    print("\n[1/4] 모델 로드 중...")
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    max_memory = {i: "70GiB" for i in range(torch.cuda.device_count())}

    # 병합 모델이 있으면 사용 (캡셔닝 학습된 모델 위에 카테고리 학습)
    use_merged = False

    if use_merged:
        print(f"병합된 모델 사용: {MERGED_MODEL_PATH}")
        model = AutoModelForImageTextToText.from_pretrained(
            MERGED_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            MERGED_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
    else:
        print(f"Base 모델 사용: {BASE_MODEL_ID}")
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=HF_CACHE_DIR,
        )
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    for param in model.parameters():
        param.requires_grad = False

    # 2. LoRA 설정
    print("\n[2/4] LoRA 설정 적용 중...")
    if args.test:
        lora_r = 8
        lora_alpha = 16
    else:
        lora_r = 16
        lora_alpha = 32

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"LoRA: r={lora_r}, alpha={lora_alpha}")

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 데이터셋 준비
    print("\n[3/4] 데이터셋 준비 중...")
    train_dataset = CategoryDataset(DATA_DIR, processor, keyword_list, max_samples=args.max_samples)
    print(f"학습 데이터: {len(train_dataset)}건")

    # 4. 학습 설정
    print("\n[4/4] 학습 시작...")
    output_dir = args.output_dir or (OUTPUT_DIR + "_test" if args.test else OUTPUT_DIR)

    logging_steps = 10 if args.test else 20
    save_steps = 50 if args.test else 100

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 8

        learning_rate=2e-4,  # 작은 데이터셋이므로 높은 학습률
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        num_train_epochs=args.epochs,

        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",

        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,

        eval_strategy="no",

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: data_collator(x, processor),
    )

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
    parser = argparse.ArgumentParser(description="Qwen3-VL 키워드 분류 LoRA 학습")

    parser.add_argument("--test", action="store_true",
                        help="테스트 모드 (100건, 1 epoch)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="학습 데이터 최대 샘플 수")
    parser.add_argument("--epochs", type=int, default=5,
                        help="학습 epoch 수 (기본: 5)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 디렉토리")

    args = parser.parse_args()

    if args.test:
        if args.max_samples is None:
            args.max_samples = 100
        if args.epochs == 5:
            args.epochs = 1

    train(args)


if __name__ == "__main__":
    main()

