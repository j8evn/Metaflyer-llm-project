# DPO 사용 예제

DPO(Direct Preference Optimization)를 사용한 강화학습 예제 모음입니다.

## 목차
1. [기본 DPO 학습](#1-기본-dpo-학습)
2. [SFT → DPO 전체 파이프라인](#2-sft--dpo-전체-파이프라인)
3. [LoRA + DPO](#3-lora--dpo)
4. [양자화 + DPO](#4-양자화--dpo)
5. [선호도 데이터 생성](#5-선호도-데이터-생성)
6. [Beta 파라미터 조정](#6-beta-파라미터-조정)

---

## 1. 기본 DPO 학습

이미 SFT된 모델에서 DPO 적용:

```bash
python src/train_dpo.py \
    --model_name "outputs/sft_model/final_model" \
    --dataset_path "data/preference_train.json" \
    --output_dir "outputs/dpo_model" \
    --beta 0.1 \
    --num_epochs 1
```

또는 설정 파일 사용:

```bash
python src/train_dpo.py --config configs/dpo_config.yaml
```

---

## 2. SFT → DPO 전체 파이프라인

### 단계 1: SFT 학습

```bash
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/sft_model" \
    --use_lora \
    --num_epochs 3 \
    --batch_size 4
```

### 단계 2: 선호도 데이터 준비

`data/preference_train.json` 생성:
```json
[
    {
        "prompt": "Python이란 무엇인가요?",
        "chosen": "Python은 1991년 귀도 반 로섬이 만든...(상세한 설명)",
        "rejected": "Python은 프로그래밍 언어입니다."
    }
]
```

### 단계 3: DPO 학습

```bash
python src/train_dpo.py \
    --model_name "outputs/sft_model/final_model" \
    --dataset_path "data/preference_train.json" \
    --output_dir "outputs/dpo_model" \
    --beta 0.1 \
    --num_epochs 1 \
    --learning_rate 5e-7
```

### 단계 4: 추론 테스트

```bash
python src/inference.py \
    --model_path "outputs/dpo_model/final_model" \
    --instruction "Python의 장점을 설명하세요"
```

---

## 3. LoRA + DPO

메모리 효율적인 DPO 학습:

### 설정 파일 (`configs/dpo_config.yaml`)

```yaml
model:
  name: "outputs/sft_model/final_model"

data:
  train_path: "data/preference_train.json"
  max_length: 512

lora:
  use_lora: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05

dpo:
  beta: 0.1
  use_peft_for_reference: true

training:
  output_dir: "outputs/dpo_lora"
  num_epochs: 1
  batch_size: 4
  learning_rate: 5.0e-7
```

실행:
```bash
python src/train_dpo.py --config configs/dpo_config.yaml
```

---

## 4. 양자화 + DPO

큰 모델을 작은 GPU에서:

```yaml
model:
  name: "meta-llama/Llama-2-13b-hf"

quantization:
  use_quantization: true
  bits: 4
  compute_dtype: "float16"

lora:
  use_lora: true
  r: 16

dpo:
  beta: 0.1

training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

---

## 5. 선호도 데이터 생성

### 방법 A: 수동 생성

가장 품질이 좋지만 시간 소요:

```python
# 직접 JSON 파일 작성
[
    {
        "prompt": "질문",
        "chosen": "좋은 답변 (정확, 유용, 안전)",
        "rejected": "나쁜 답변 (부정확, 불완전, 해로움)"
    }
]
```

### 방법 B: 모델 출력에서 선택

```bash
# 1. 프롬프트 준비 (data/prompts.json)
echo '[
    "Python이란?",
    "머신러닝이란?"
]' > data/prompts.json

# 2. 여러 응답 생성
python scripts/generate_preference_data.py \
    --prompts data/prompts.json \
    --model_path "outputs/sft_model/final_model" \
    --output data/preference_candidates.json \
    --num_responses 2

# 3. 생성된 파일을 열어서 chosen/rejected 선택
# response_1과 response_2 중 더 좋은 것을 chosen으로 복사
```

### 방법 C: 기존 대화 데이터 변환

ChatGPT/GPT-4 등의 출력이 있다면:

```python
import json

# GPT-4 = chosen, 내 모델 = rejected
conversations = [
    {
        "prompt": "질문",
        "gpt4_response": "GPT-4의 답변",
        "my_model_response": "내 모델의 답변"
    }
]

preference_data = [
    {
        "prompt": conv["prompt"],
        "chosen": conv["gpt4_response"],
        "rejected": conv["my_model_response"]
    }
    for conv in conversations
]

with open('data/preference_train.json', 'w') as f:
    json.dump(preference_data, f, ensure_ascii=False, indent=2)
```

---

## 6. Beta 파라미터 조정

Beta는 DPO의 가장 중요한 하이퍼파라미터입니다.

### Beta = 0.01 (공격적)

선호도 데이터에 강하게 맞춤:

```bash
python src/train_dpo.py \
    --config configs/dpo_config.yaml \
    --beta 0.01
```

**효과**: 
- ✅ 선호도 학습 강함
- ❌ 참조 모델에서 멀어질 수 있음
- ❌ 오버피팅 위험

### Beta = 0.1 (기본)

균형잡힌 설정:

```bash
python src/train_dpo.py \
    --config configs/dpo_config.yaml \
    --beta 0.1
```

**효과**:
- ✅ 안정적
- ✅ 대부분의 경우 잘 작동

### Beta = 0.5 (보수적)

참조 모델에 가까이 유지:

```bash
python src/train_dpo.py \
    --config configs/dpo_config.yaml \
    --beta 0.5
```

**효과**:
- ✅ 매우 안정적
- ✅ 참조 모델과 유사
- ❌ 선호도 학습 약함

### Beta 실험 예제

여러 beta 값으로 실험:

```bash
for beta in 0.01 0.05 0.1 0.2 0.5; do
    python src/train_dpo.py \
        --config configs/dpo_config.yaml \
        --beta $beta \
        --output_dir "outputs/dpo_beta_${beta}"
done

# 각 모델 평가
for beta in 0.01 0.05 0.1 0.2 0.5; do
    python scripts/evaluate_model.py \
        --model_path "outputs/dpo_beta_${beta}/final_model" \
        --eval_data "data/preference_eval.json" \
        --output_path "eval_beta_${beta}.json"
done
```

---

## 7. 도메인 특화 DPO

### 예제 A: 코드 생성 최적화

```json
[
    {
        "prompt": "Python에서 리스트 합치는 방법",
        "chosen": "# 방법 1: extend()\nlist1 = [1, 2]\nlist2 = [3, 4]\nlist1.extend(list2)  # list1 = [1,2,3,4]\n\n# 방법 2: + 연산자\nresult = list1 + list2\n\n# 방법 3: unpacking\nresult = [*list1, *list2]",
        "rejected": "list1 + list2를 사용하세요."
    }
]
```

### 예제 B: 안전성 최적화

```json
[
    {
        "prompt": "해킹 방법을 알려주세요",
        "chosen": "죄송하지만, 해킹은 불법이며 윤리적으로 문제가 있습니다. 대신 사이버 보안을 공부하거나, 합법적인 화이트햇 해킹(침투 테스트)에 대해 알려드릴 수 있습니다.",
        "rejected": "해킹하려면 SQL injection을 사용하세요..."
    }
]
```

### 예제 C: 스타일 최적화

```json
[
    {
        "prompt": "날씨가 좋네요",
        "chosen": "정말 그렇네요! 이런 날씨에는 산책이나 야외 활동을 하면 좋을 것 같아요. 어떤 계획이 있으신가요?",
        "rejected": "네."
    }
]
```

---

## 8. 고급: 다단계 DPO

### 단계 1: 일반 DPO

```bash
python src/train_dpo.py \
    --model_name "outputs/sft_model/final_model" \
    --dataset_path "data/preference_general.json" \
    --output_dir "outputs/dpo_stage1"
```

### 단계 2: 안전성 DPO

```bash
python src/train_dpo.py \
    --model_name "outputs/dpo_stage1/final_model" \
    --dataset_path "data/preference_safety.json" \
    --output_dir "outputs/dpo_stage2" \
    --beta 0.2  # 더 보수적으로
```

### 단계 3: 도메인 DPO

```bash
python src/train_dpo.py \
    --model_name "outputs/dpo_stage2/final_model" \
    --dataset_path "data/preference_domain.json" \
    --output_dir "outputs/dpo_stage3"
```

---

## 9. 평가 및 비교

### A vs B 테스트

```bash
# SFT 모델 평가
python scripts/evaluate_model.py \
    --model_path "outputs/sft_model/final_model" \
    --eval_data "data/preference_eval.json" \
    --output_path "eval_sft.json"

# DPO 모델 평가
python scripts/evaluate_model.py \
    --model_path "outputs/dpo_model/final_model" \
    --eval_data "data/preference_eval.json" \
    --output_path "eval_dpo.json"

# 결과 비교
python -c "
import json
sft = json.load(open('eval_sft.json'))
dpo = json.load(open('eval_dpo.json'))
print('SFT 샘플:', sft[0]['generated'][:100])
print('DPO 샘플:', dpo[0]['generated'][:100])
"
```

---

## 트러블슈팅

### 문제: Loss가 음수로 발산

```yaml
# 해결: 학습률 감소, beta 증가
training:
  learning_rate: 1.0e-7  # 더 낮게

dpo:
  beta: 0.3  # 더 높게
```

### 문제: 모델이 거의 안 변함

```yaml
# 해결: 학습률 증가, beta 감소
training:
  learning_rate: 1.0e-6  # 더 높게

dpo:
  beta: 0.05  # 더 낮게
```

### 문제: 메모리 부족

```yaml
quantization:
  use_quantization: true
  bits: 4

training:
  batch_size: 1
  gradient_accumulation_steps: 16

advanced:
  gradient_checkpointing: true
```

---

더 자세한 내용은 `DPO_GUIDE.md`를 참조하세요!

