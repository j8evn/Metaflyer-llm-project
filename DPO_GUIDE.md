# DPO (Direct Preference Optimization) 가이드

DPO는 RLHF(Reinforcement Learning from Human Feedback)의 효율적인 대안으로, 선호도 데이터를 사용하여 모델을 직접 최적화하는 방법입니다.

## 목차
1. [DPO란?](#dpo란)
2. [선호도 데이터 준비](#선호도-데이터-준비)
3. [DPO 학습 실행](#dpo-학습-실행)
4. [설정 옵션](#설정-옵션)
5. [베스트 프랙티스](#베스트-프랙티스)

---

## DPO란?

### RLHF vs DPO

**전통적인 RLHF**:
1. Supervised Fine-tuning (SFT)
2. Reward Model 학습
3. PPO로 정책 최적화

**DPO**:
1. Supervised Fine-tuning (SFT)
2. DPO로 직접 최적화 (한 단계!)

### DPO의 장점

✅ **간단함**: Reward Model과 PPO 없이 한 번에 학습  
✅ **효율적**: 메모리와 시간 절약  
✅ **안정적**: PPO보다 학습이 안정적  
✅ **성능**: RLHF와 유사한 성능

---

## 선호도 데이터 준비

### 데이터 형식

선호도 데이터는 다음 형식이어야 합니다:

```json
[
    {
        "prompt": "질문 또는 지시사항",
        "chosen": "더 좋은 응답",
        "rejected": "덜 좋은 응답"
    }
]
```

### 방법 1: 수동으로 데이터 생성

```json
[
    {
        "prompt": "Python에서 리스트와 튜플의 차이점을 설명하세요.",
        "chosen": "리스트는 변경 가능한(mutable) 자료구조로...(상세한 설명)",
        "rejected": "리스트는 []를 사용하고 튜플은 ()를 사용합니다."
    }
]
```

### 방법 2: Instruction 형식 프롬프트

```json
[
    {
        "prompt": {
            "instruction": "다음 문장을 번역하세요.",
            "input": "Hello, world!"
        },
        "chosen": "안녕하세요, 세계!",
        "rejected": "헬로 월드"
    }
]
```

### 방법 3: 기존 모델로 응답 생성 후 선택

```bash
# 1. 프롬프트 파일 준비 (prompts.json)
# 2. 모델로 여러 응답 생성
python scripts/generate_preference_data.py \
    --prompts data/prompts.json \
    --model_path "your-model-path" \
    --output data/preference_candidates.json

# 3. 생성된 응답 중 chosen/rejected 선택
# (수동으로 편집)
```

### 샘플 데이터 생성

```bash
# 프로젝트에 포함된 샘플 생성
python -c "from src.dpo_utils import create_preference_sample_dataset; \
create_preference_sample_dataset('data/preference_train.json', 50); \
create_preference_sample_dataset('data/preference_eval.json', 10)"
```

---

## DPO 학습 실행

### 1단계: SFT 모델 준비

DPO는 이미 파인튜닝된 모델에서 시작합니다:

```bash
# SFT 먼저 실행
python src/train.py --config configs/train_config.yaml
```

### 2단계: DPO 학습

```bash
# 기본 DPO 학습
python src/train_dpo.py --config configs/dpo_config.yaml
```

### 커맨드 라인으로 실행

```bash
python src/train_dpo.py \
    --model_name "your-sft-model-path" \
    --dataset_path "data/preference_train.json" \
    --output_dir "outputs/dpo_model" \
    --num_epochs 1 \
    --batch_size 4 \
    --learning_rate 5e-7 \
    --beta 0.1
```

---

## 설정 옵션

### DPO 핵심 파라미터

#### `beta` (기본값: 0.1)
DPO 손실 함수의 온도 파라미터:
- **높은 값 (0.5)**: 참조 모델에 가까이 유지 (보수적)
- **낮은 값 (0.01)**: 선호도 데이터에 더 집중 (공격적)
- **권장**: 0.1 ~ 0.5

```yaml
dpo:
  beta: 0.1
```

#### `loss_type` (기본값: "sigmoid")
손실 함수 타입:
- **sigmoid**: 표준 DPO 손실 (기본)
- **hinge**: Hinge 손실
- **ipo**: Identity Preference Optimization
- **kto_pair**: Kahneman-Tversky Optimization

```yaml
dpo:
  loss_type: "sigmoid"
```

#### `learning_rate` (기본값: 5e-7)
DPO는 SFT보다 낮은 학습률 사용:
- **권장**: 1e-7 ~ 1e-6
- SFT 학습률의 1/10 ~ 1/100

```yaml
training:
  learning_rate: 5.0e-7
```

#### `num_epochs` (기본값: 1)
DPO는 보통 1-2 에포크면 충분:
- **1 에포크**: 대부분의 경우 충분
- **2-3 에포크**: 데이터가 많거나 복잡한 경우

```yaml
training:
  num_epochs: 1
```

### 완전한 설정 예제

```yaml
# configs/dpo_config.yaml

model:
  name: "outputs/checkpoints/final_model"  # SFT 모델 경로

data:
  train_path: "data/preference_train.json"
  eval_path: "data/preference_eval.json"
  max_length: 512
  max_prompt_length: 256

dpo:
  beta: 0.1
  loss_type: "sigmoid"
  use_peft_for_reference: true

training:
  output_dir: "outputs/dpo_checkpoints"
  num_epochs: 1
  batch_size: 4
  learning_rate: 5.0e-7

lora:
  use_lora: true
  r: 16
  lora_alpha: 32
```

---

## 베스트 프랙티스

### 1. 데이터 품질이 핵심

✅ **좋은 선호도 데이터**:
- Chosen 응답: 정확하고, 유용하고, 안전한 답변
- Rejected 응답: 부정확하거나, 불완전하거나, 해로운 답변
- 명확한 차이가 있어야 함

❌ **피해야 할 것**:
- Chosen과 Rejected가 거의 비슷함
- Chosen이 실제로 더 나쁜 경우
- 주관적이고 애매한 선호도

### 2. SFT 먼저, DPO는 나중에

올바른 순서:
```
베이스 모델 → SFT → DPO → 최종 모델
```

### 3. 적절한 데이터 양

- **최소**: 100-500 샘플
- **권장**: 1,000-10,000 샘플
- **최적**: 10,000+ 샘플

### 4. 하이퍼파라미터 튜닝

시작점:
```yaml
learning_rate: 5.0e-7
beta: 0.1
num_epochs: 1
batch_size: 4
```

조정 전략:
1. 기본 설정으로 시작
2. 검증 세트로 평가
3. Beta 값 조정 (0.05 ~ 0.5)
4. 필요시 학습률 미세 조정

### 5. LoRA와 함께 사용

DPO + LoRA는 환상의 조합:

```yaml
lora:
  use_lora: true
  r: 16
  lora_alpha: 32

quantization:
  use_quantization: true
  bits: 4
```

메모리 절약 + 효율적 학습!

### 6. 평가 및 모니터링

학습 중 모니터링:
```bash
tensorboard --logdir outputs/dpo_checkpoints/logs
```

평가 메트릭:
- Loss 감소 추이
- 검증 세트 손실
- 실제 응답 품질 (수동 평가)

---

## 전체 워크플로우 예제

### 단계별 가이드

```bash
# 1. 환경 설정
./setup.sh

# 2. SFT 데이터 준비
# data/train.json 준비

# 3. SFT 학습
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/sft_model" \
    --use_lora \
    --num_epochs 3

# 4. 선호도 데이터 준비
# data/preference_train.json 준비

# 5. DPO 학습
python src/train_dpo.py \
    --model_name "outputs/sft_model/final_model" \
    --dataset_path "data/preference_train.json" \
    --output_dir "outputs/dpo_model" \
    --beta 0.1 \
    --num_epochs 1

# 6. 추론 테스트
python src/inference.py \
    --model_path "outputs/dpo_model/final_model" \
    --instruction "Python이란 무엇인가요?"
```

---

## 고급 팁

### 1. 다단계 DPO

여러 번의 DPO 반복:
```
SFT → DPO (일반) → DPO (안전성) → DPO (스타일)
```

### 2. 도메인 특화

특정 도메인의 선호도 학습:
- 의료: 안전하고 정확한 답변 선호
- 코딩: 실행 가능하고 효율적인 코드 선호
- 창의적 글쓰기: 흥미롭고 독창적인 내용 선호

### 3. Beta 값 스케줄링

학습 중 beta 값 조정:
- 초반: 높은 beta (안정적)
- 후반: 낮은 beta (선호도 강화)

---

## 트러블슈팅

### 문제 1: Loss가 감소하지 않음

해결책:
- 학습률 감소 (5e-7 → 1e-7)
- Beta 값 조정
- 데이터 품질 확인

### 문제 2: 모델이 참조 모델과 너무 다름

해결책:
- Beta 값 증가 (0.1 → 0.5)
- 에포크 수 감소
- 학습률 감소

### 문제 3: GPU 메모리 부족

해결책:
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16

lora:
  use_lora: true

quantization:
  use_quantization: true
  bits: 4
```

---

## 참고 자료

- 논문: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- TRL 라이브러리: https://github.com/huggingface/trl
- Hugging Face 가이드: https://huggingface.co/docs/trl/dpo_trainer

더 많은 예제는 `EXAMPLES.md`를 참조하세요!

