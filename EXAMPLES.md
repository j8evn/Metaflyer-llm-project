# 사용 예제

이 문서는 LLM 파인튜닝 프로젝트의 다양한 사용 예제를 제공합니다.

## 목차
1. [기본 파인튜닝](#1-기본-파인튜닝)
2. [LoRA를 사용한 효율적 파인튜닝](#2-lora를-사용한-효율적-파인튜닝)
3. [양자화를 사용한 메모리 효율적 학습](#3-양자화를-사용한-메모리-효율적-학습)
4. [커스텀 데이터셋 사용](#4-커스텀-데이터셋-사용)
5. [추론 및 테스트](#5-추론-및-테스트)
6. [모델 평가](#6-모델-평가)
7. [LoRA 가중치 병합](#7-lora-가중치-병합)
8. [모델 양자화](#8-모델-양자화)

---

## 1. 기본 파인튜닝

가장 간단한 파인튜닝 방법입니다.

### 예제 1-1: GPT-2 모델 파인튜닝

```bash
python src/train.py \
    --model_name "gpt2" \
    --dataset_path "data/train.json" \
    --output_dir "models/gpt2-finetuned" \
    --num_epochs 3 \
    --batch_size 4
```

### 예제 1-2: 설정 파일 사용

```bash
python src/train.py --config configs/train_config.yaml
```

---

## 2. LoRA를 사용한 효율적 파인튜닝

LoRA는 전체 모델이 아닌 일부 파라미터만 학습하여 메모리를 크게 절약합니다.

### 예제 2-1: Llama-2 7B with LoRA

```bash
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/train.json" \
    --output_dir "models/llama2-lora" \
    --use_lora \
    --num_epochs 3 \
    --batch_size 4
```

### 예제 2-2: 커스텀 LoRA 설정

`configs/train_config.yaml`에서 LoRA 설정을 조정:

```yaml
lora:
  use_lora: true
  r: 16                # LoRA rank (높을수록 용량 증가)
  lora_alpha: 32       # LoRA alpha
  lora_dropout: 0.05
  target_modules:      # LoRA 적용 모듈
    - "q_proj"
    - "v_proj"
```

---

## 3. 양자화를 사용한 메모리 효율적 학습

4bit 또는 8bit 양자화로 큰 모델을 작은 GPU에서 학습할 수 있습니다.

### 예제 3-1: 4-bit 양자화 + LoRA

`configs/train_config.yaml` 수정:

```yaml
quantization:
  use_quantization: true
  bits: 4
  compute_dtype: "float16"

lora:
  use_lora: true
  r: 16
  lora_alpha: 32
```

실행:
```bash
python src/train.py --config configs/train_config.yaml
```

### 예제 3-2: 8-bit 양자화

```yaml
quantization:
  use_quantization: true
  bits: 8
```

---

## 4. 커스텀 데이터셋 사용

### 예제 4-1: 샘플 데이터 생성

```bash
# Instruction 형식 데이터 생성
python scripts/create_sample_data.py --format instruction --num_train 100

# Chat 형식 데이터 생성
python scripts/create_sample_data.py --format chat --num_train 50

# 모든 형식 생성
python scripts/create_sample_data.py --format all
```

### 예제 4-2: 자신의 데이터셋 준비

**Instruction 형식 (`data/my_data.json`):**

```json
[
    {
        "instruction": "다음 텍스트를 요약하세요.",
        "input": "긴 텍스트...",
        "output": "요약된 텍스트..."
    }
]
```

학습:
```bash
python src/train.py \
    --dataset_path "data/my_data.json" \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --output_dir "models/my-model"
```

---

## 5. 추론 및 테스트

### 예제 5-1: 대화형 모드

```bash
python src/inference.py --model_path "models/llama2-finetuned"
```

그러면 대화형 인터페이스가 시작됩니다:
```
질문을 입력하세요: Python이란 무엇인가요?
응답:
Python은...
```

### 예제 5-2: 단일 프롬프트

```bash
python src/inference.py \
    --model_path "models/llama2-finetuned" \
    --prompt "Hello, how are you?"
```

### 예제 5-3: Instruction 형식

```bash
python src/inference.py \
    --model_path "models/llama2-finetuned" \
    --instruction "다음 코드를 설명하세요" \
    --input "print('Hello World')" \
    --max_new_tokens 256
```

### 예제 5-4: 양자화로 로딩 (메모리 절약)

```bash
python src/inference.py \
    --model_path "models/llama2-finetuned" \
    --load_in_8bit
```

---

## 6. 모델 평가

### 예제 6-1: 평가 데이터셋으로 평가

```bash
python scripts/evaluate_model.py \
    --model_path "models/llama2-finetuned" \
    --eval_data "data/eval.json" \
    --output_path "evaluation_results.json"
```

### 예제 6-2: 일부 샘플만 평가

```bash
python scripts/evaluate_model.py \
    --model_path "models/llama2-finetuned" \
    --eval_data "data/eval.json" \
    --max_samples 10 \
    --output_path "quick_eval.json"
```

---

## 7. LoRA 가중치 병합

LoRA 어댑터를 베이스 모델과 병합하여 단일 모델로 만듭니다.

```bash
python scripts/convert_checkpoint.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --lora_model "models/llama2-lora" \
    --output "models/llama2-merged"
```

병합 후에는 일반 모델처럼 사용 가능:
```bash
python src/inference.py --model_path "models/llama2-merged"
```

---

## 8. 모델 양자화

학습 완료 후 모델을 양자화하여 크기를 줄입니다.

### 예제 8-1: 8-bit 양자화

```bash
python scripts/quantize_model.py \
    --model_path "models/llama2-finetuned" \
    --output "models/llama2-finetuned-8bit" \
    --bits 8
```

### 예제 8-2: 4-bit 양자화

```bash
python scripts/quantize_model.py \
    --model_path "models/llama2-finetuned" \
    --output "models/llama2-finetuned-4bit" \
    --bits 4
```

---

## 고급 예제

### 예제 9: 멀티 GPU 학습

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py \
    --config configs/train_config.yaml
```

### 예제 10: WandB로 학습 모니터링

`configs/train_config.yaml`:
```yaml
monitoring:
  use_wandb: true
  wandb_project: "my-llm-project"
```

실행 전 WandB 로그인:
```bash
wandb login
python src/train.py --config configs/train_config.yaml
```

### 예제 11: 그래디언트 체크포인팅으로 메모리 절약

`configs/train_config.yaml`:
```yaml
advanced:
  gradient_checkpointing: true
  gradient_accumulation_steps: 8
```

---

## 팁과 트릭

### GPU 메모리가 부족할 때

1. **LoRA 사용** - 메모리 사용량 크게 감소
2. **양자화 활성화** - 4bit 또는 8bit
3. **배치 크기 감소** - `batch_size: 1` 또는 2
4. **그래디언트 누적 사용** - `gradient_accumulation_steps: 8`
5. **그래디언트 체크포인팅** - `gradient_checkpointing: true`
6. **max_length 감소** - `max_length: 256` 또는 512

### 학습 속도 향상

1. **적절한 배치 크기** - GPU 메모리가 허용하는 최대치
2. **mixed precision** - `bf16: true` (A100/H100) 또는 `fp16: true`
3. **데이터 로더 워커** - `dataloader_num_workers: 4`
4. **효율적인 옵티마이저** - `adamw_8bit` 사용

### 더 나은 결과를 위해

1. **학습률 조정** - `learning_rate: 2e-5` (작은 값부터 시작)
2. **충분한 에포크** - 3-5 에포크
3. **워밍업 스텝** - `warmup_steps: 100`
4. **데이터 품질** - 고품질의 학습 데이터가 가장 중요
5. **평가 및 검증** - 정기적으로 검증 세트로 평가

---

## 트러블슈팅

### CUDA out of memory

```bash
# 해결책 1: 배치 크기 감소
--batch_size 1

# 해결책 2: LoRA + 양자화 사용
# configs/train_config.yaml에서 설정

# 해결책 3: 더 작은 모델 사용
--model_name "gpt2"  # 대신 "gpt2-xl"
```

### 모델 로딩 실패

```bash
# Hugging Face 토큰이 필요한 모델의 경우
huggingface-cli login

# 그리고 다시 시도
python src/train.py --config configs/train_config.yaml
```

더 많은 도움말은 [README.md](README.md)를 참조하세요!

