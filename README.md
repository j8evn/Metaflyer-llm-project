# LLM 파인튜닝 및 재학습 프로젝트

오픈 소스 LLM 모델의 파인튜닝과 재학습을 위한 Python 프로젝트입니다.

## 주요 기능

- 🚀 **다양한 오픈 소스 LLM 지원**: Llama, Mistral, GPT-2, BERT 등
- 🎯 **효율적인 파인튜닝**: LoRA, QLoRA를 활용한 메모리 효율적 학습
- 🏆 **DPO 강화학습**: Direct Preference Optimization으로 RLHF 대체
- 🌐 **REST API 서버**: FastAPI 기반 프로덕션 레디 API
- 📊 **데이터 전처리**: 커스텀 데이터셋 처리 및 포맷팅
- 🔧 **유연한 설정**: YAML 기반 설정 관리
- 📈 **학습 모니터링**: WandB, TensorBoard 지원

## 프로젝트 구조

```
llm/
├── configs/          # 설정 파일들
│   ├── train_config.yaml  # SFT 학습 설정
│   └── dpo_config.yaml    # DPO 학습 설정
├── data/             # 학습 데이터
├── models/           # 저장된 모델
├── outputs/          # 학습 결과 및 로그
├── src/              # 소스 코드
│   ├── data_utils.py      # 데이터 처리
│   ├── dpo_utils.py       # DPO 데이터 처리
│   ├── model_utils.py     # 모델 로딩 및 설정
│   ├── train.py           # SFT 학습 스크립트
│   ├── train_dpo.py       # DPO 학습 스크립트
│   └── inference.py       # 추론 스크립트
├── scripts/          # 유틸리티 스크립트
├── notebooks/        # Jupyter 노트북
├── requirements.txt  # 의존성 패키지
├── README.md         # 프로젝트 문서
└── DPO_GUIDE.md      # DPO 가이드
```

## 설치 방법

1. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 기본 파인튜닝

```bash
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/train.json" \
    --output_dir "models/llama2-finetuned" \
    --num_epochs 3 \
    --batch_size 4
```

### 2. LoRA를 사용한 효율적 파인튜닝

```bash
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/train.json" \
    --output_dir "models/llama2-lora" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epochs 3
```

### 3. 설정 파일을 사용한 학습

```bash
python src/train.py --config configs/train_config.yaml
```

### 4. DPO 강화학습 (선택사항)

```bash
python src/train_dpo.py \
    --model_name "models/llama2-finetuned" \
    --dataset_path "data/preference_train.json" \
    --output_dir "models/llama2-dpo" \
    --beta 0.1 \
    --num_epochs 1
```

### 5. 추론 실행

```bash
python src/inference.py \
    --model_path "models/llama2-dpo" \
    --prompt "당신의 질문을 입력하세요"
```

### 6. API 서버 실행

```bash
# API 서버 시작
python src/api_server.py \
    --model_path "models/llama2-dpo" \
    --host 0.0.0.0 \
    --port 8000

# 또는 스크립트 사용
./scripts/start_api.sh

# API 문서: http://localhost:8000/docs
```

API 사용 예제:
```bash
# cURL로 요청
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Python이란?", "max_new_tokens": 200}'
```

## 데이터 포맷

### SFT (Supervised Fine-Tuning) 데이터

```json
[
    {
        "instruction": "질문 또는 지시사항",
        "input": "추가 입력 (선택사항)",
        "output": "기대되는 출력"
    }
]
```

### DPO (선호도) 데이터

```json
[
    {
        "prompt": "질문 또는 프롬프트",
        "chosen": "더 좋은 응답",
        "rejected": "덜 좋은 응답"
    }
]
```

자세한 내용은 `DPO_GUIDE.md`를 참조하세요.

## 설정 옵션

주요 설정 파라미터:

- `model_name`: 사용할 모델 이름 (Hugging Face model ID)
- `dataset_path`: 학습 데이터 경로
- `output_dir`: 모델 저장 디렉토리
- `num_epochs`: 학습 에포크 수
- `batch_size`: 배치 크기
- `learning_rate`: 학습률
- `use_lora`: LoRA 사용 여부
- `quantization`: 양자화 옵션 (4bit, 8bit)

자세한 설정은 `configs/train_config.yaml`을 참조하세요.

## 지원 모델

- Llama 2/3 (Meta)
- Mistral (Mistral AI)
- Falcon (TII)
- GPT-2 (OpenAI)
- BLOOM (BigScience)
- 기타 Hugging Face Transformers 호환 모델

## 학습 모니터링

### WandB 사용
```bash
export WANDB_PROJECT="llm-finetuning"
python src/train.py --config configs/train_config.yaml
```

### TensorBoard 사용
```bash
tensorboard --logdir outputs/logs
```

## 메모리 최적화 팁

1. **LoRA 사용**: 전체 파인튜닝 대비 메모리 사용량 크게 감소
2. **그래디언트 체크포인팅**: `gradient_checkpointing=True` 설정
3. **4bit 양자화**: `quantization=4bit` 설정
4. **배치 크기 조정**: GPU 메모리에 맞게 조정
5. **그래디언트 누적**: `gradient_accumulation_steps` 사용

## 라이선스

MIT License

## 기여

이슈와 풀 리퀘스트를 환영합니다!

