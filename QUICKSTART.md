# 빠른 시작 가이드

이 가이드는 프로젝트를 빠르게 시작하는 방법을 안내합니다.

## 1. 환경 설정 (5분)

### 방법 A: 자동 설정 스크립트 사용 (추천)

```bash
cd /Users/jerry/metaflyer/llm
chmod +x setup.sh
./setup.sh
```

이 스크립트는 자동으로:
- 가상환경 생성
- 의존성 패키지 설치
- 샘플 데이터 생성

### 방법 B: 수동 설정

```bash
# 1. 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 샘플 데이터 생성
python src/data_utils.py
```

## 2. 첫 번째 파인튜닝 (10분)

### 옵션 1: 작은 모델로 빠른 테스트 (GPT-2)

```bash
python src/train.py \
    --model_name "gpt2" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/gpt2-test" \
    --num_epochs 1 \
    --batch_size 2
```

**예상 시간**: 5-10분 (GPU), 30-60분 (CPU)

### 옵션 2: 큰 모델 + LoRA (권장)

```bash
python src/train.py --config configs/train_config.yaml
```

설정 파일을 먼저 수정하세요:
```yaml
# configs/train_config.yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # 또는 다른 모델

lora:
  use_lora: true  # LoRA 활성화 (메모리 절약)

training:
  num_epochs: 3
  batch_size: 4
```

**예상 시간**: 1-3시간 (GPU 사양에 따라)

## 3. 모델 추론 테스트

학습이 완료되면 바로 테스트해보세요:

```bash
# 대화형 모드
python src/inference.py --model_path "outputs/checkpoints/final_model"

# 또는 직접 질문
python src/inference.py \
    --model_path "outputs/checkpoints/final_model" \
    --instruction "Python이란 무엇인가요?"
```

## 4. 다음 단계

### 자신의 데이터로 학습하기

1. **데이터 준비** - `data/my_data.json` 형식:
   ```json
   [
       {
           "instruction": "질문 또는 지시사항",
           "input": "추가 입력 (선택)",
           "output": "기대되는 답변"
       }
   ]
   ```

2. **설정 수정** - `configs/train_config.yaml`:
   ```yaml
   data:
     train_path: "data/my_data.json"
   
   model:
     name: "원하는 모델 이름"
   ```

3. **학습 실행**:
   ```bash
   python src/train.py --config configs/train_config.yaml
   ```

### 성능 개선하기

1. **하이퍼파라미터 조정**:
   - `learning_rate`: 2e-5 ~ 5e-5
   - `num_epochs`: 3 ~ 5
   - `lora.r`: 8, 16, 32 (높을수록 용량/성능 증가)

2. **더 많은 데이터**: 일반적으로 1000+ 샘플 권장

3. **모델 평가**:
   ```bash
   python scripts/evaluate_model.py \
       --model_path "outputs/checkpoints/final_model" \
       --eval_data "data/eval.json"
   ```

## 주요 명령어 요약

```bash
# 학습
python src/train.py --config configs/train_config.yaml

# 추론
python src/inference.py --model_path "모델경로"

# 평가
python scripts/evaluate_model.py --model_path "모델경로" --eval_data "data/eval.json"

# 샘플 데이터 생성
python scripts/create_sample_data.py --format all

# LoRA 병합
python scripts/convert_checkpoint.py --base_model "베이스모델" --lora_model "LoRA모델" --output "출력경로"
```

## GPU 메모리 부족 시

```bash
# 1. LoRA 사용 (configs/train_config.yaml)
lora:
  use_lora: true

# 2. 양자화 사용
quantization:
  use_quantization: true
  bits: 4

# 3. 배치 크기 감소
training:
  batch_size: 1
  gradient_accumulation_steps: 8
```

## 도움말

- 전체 문서: [README.md](README.md)
- 상세 예제: [EXAMPLES.md](EXAMPLES.md)
- Jupyter 노트북: `notebooks/quickstart.ipynb`

## 문제 해결

### CUDA out of memory
→ 배치 크기 감소, LoRA 사용, 양자화 활성화

### 모델 로딩 실패
→ Hugging Face 계정 로그인 필요: `huggingface-cli login`

### 느린 학습 속도
→ GPU 사용 확인, 배치 크기 증가, mixed precision 활성화

더 많은 정보는 `README.md`와 `EXAMPLES.md`를 참조하세요!

