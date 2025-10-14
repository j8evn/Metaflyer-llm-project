# 새 모델 추가하기 - 실전 가이드

새로운 LLM 모델을 프로젝트에 추가하는 실전 가이드입니다.

## 🎯 90%의 경우: 아무것도 안 해도 됩니다!

대부분의 Hugging Face 모델은 **즉시 사용 가능**합니다:

```bash
python src/train.py --model_name "새로운-모델-이름"
```

끝! 🎉

---

## 📝 3단계로 새 모델 추가하기

### 1단계: 모델 호환성 확인 (1분)

```bash
# 빠른 체크 (토크나이저만)
python scripts/check_model_compatibility.py "mistralai/Mistral-7B-v0.1" --quick

# 완전한 체크 (모델 로딩 포함)
python scripts/check_model_compatibility.py "mistralai/Mistral-7B-v0.1"
```

### 2단계: 테스트 학습 (5-10분)

```bash
python src/train.py \
    --model_name "mistralai/Mistral-7B-v0.1" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/test_mistral" \
    --num_epochs 1 \
    --batch_size 2
```

### 3단계: 추론 테스트 (1분)

```bash
python src/inference.py \
    --model_path "outputs/test_mistral/final_model" \
    --instruction "테스트 질문"
```

---

## 🔧 특수 설정이 필요한 모델

### Qwen 모델

```yaml
# configs/train_config.yaml
model:
  name: "Qwen/Qwen-7B"
  trust_remote_code: true  # 필수!
```

실행:
```bash
python src/train.py --config configs/train_config.yaml
```

### Gemma 모델

```bash
# 1. Hugging Face 로그인 (필수)
huggingface-cli login

# 2. 모델 페이지에서 라이선스 동의
# https://huggingface.co/google/gemma-7b

# 3. 학습
python src/train.py --model_name "google/gemma-7b"
```

### Falcon 모델

```yaml
# configs/train_config.yaml
model:
  name: "tiiuae/falcon-7b"
  trust_remote_code: true  # 필수!
```

---

## 📋 지원 모델 전체 목록

### 즉시 사용 가능 (50+ 모델)

#### 7B 클래스
- `meta-llama/Llama-2-7b-hf` - Meta AI
- `mistralai/Mistral-7B-v0.1` - Mistral AI
- `google/gemma-7b` - Google (인증 필요)
- `Qwen/Qwen-7B` - Alibaba (trust_remote_code)
- `tiiuae/falcon-7b` - TII (trust_remote_code)
- `01-ai/Yi-6B` - 01.AI
- `stabilityai/stablelm-3b-4e1t` - Stability AI

#### 소형 모델 (테스트용)
- `gpt2` - 124M (가장 빠름)
- `gpt2-medium` - 355M
- `gpt2-large` - 774M
- `gpt2-xl` - 1.5B
- `microsoft/phi-2` - 2.7B

#### 대형 모델
- `meta-llama/Llama-2-13b-hf` - 13B
- `meta-llama/Llama-2-70b-hf` - 70B
- `mistralai/Mixtral-8x7B-v0.1` - 8x7B MoE

### 사용 예제

```bash
# GPT-2 (빠른 테스트)
python src/train.py --model_name "gpt2"

# Mistral 7B
python src/train.py --model_name "mistralai/Mistral-7B-v0.1"

# Qwen 7B
python src/train.py \
    --model_name "Qwen/Qwen-7B" \
    # configs/train_config.yaml에서 trust_remote_code: true 설정

# Gemma 7B
huggingface-cli login
python src/train.py --model_name "google/gemma-7b"
```

---

## 🚀 실전: 새 모델 추가 워크플로우

### 예시: Yi-34B 모델 추가

```bash
# 1. 호환성 확인
python scripts/check_model_compatibility.py "01-ai/Yi-34B" --quick

# 2. 설정 파일 생성
cat > configs/yi_config.yaml << 'YAML'
model:
  name: "01-ai/Yi-34B"

data:
  train_path: "data/train.json"
  max_length: 4096

lora:
  use_lora: true
  r: 16
  lora_alpha: 32

quantization:
  use_quantization: true  # 34B 모델은 양자화 권장
  bits: 4

training:
  num_epochs: 3
  batch_size: 1  # 큰 모델은 작은 배치
  gradient_accumulation_steps: 16
YAML

# 3. 학습 실행
python src/train.py --config configs/yi_config.yaml

# 4. 추론 테스트
python src/inference.py \
    --model_path "outputs/checkpoints/final_model" \
    --load_in_4bit
```

---

## 🔍 모델 선택 가이드

### 모델 크기별 추천

| 크기 | 모델 | 용도 | GPU 메모리 (LoRA) |
|------|------|------|-------------------|
| ~1B | GPT-2, Phi-2 | 테스트, 실험 | 4GB |
| 3-7B | Mistral-7B, Gemma-7B | 일반 용도 | 16GB |
| 13B | Llama-2-13B | 고성능 | 24GB |
| 34B+ | Yi-34B, Mixtral | 최고 성능 | 40GB+ (4bit) |

### 라이선스별 분류

**상업적 사용 가능:**
- ✅ Mistral (Apache 2.0)
- ✅ Falcon (Apache 2.0)
- ✅ Yi (Apache 2.0)
- ✅ GPT-2 (MIT)

**제한적 라이선스:**
- ⚠️ Llama 2 (Llama 2 Community License)
- ⚠️ Gemma (Gemma Terms of Use)
- ⚠️ Qwen (Tongyi Qianwen License)

---

## 💻 코드 예제

### 여러 모델로 자동 벤치마크

```python
# scripts/benchmark_models.py
"""
여러 모델 성능 비교
"""

import subprocess
import json

# 테스트할 모델 목록
MODELS = [
    {"name": "gpt2", "batch_size": 8},
    {"name": "microsoft/phi-2", "batch_size": 4},
    {"name": "mistralai/Mistral-7B-v0.1", "batch_size": 4},
]

results = []

for model_info in MODELS:
    model_name = model_info["name"]
    batch_size = model_info["batch_size"]
    
    print(f"\n{'='*60}")
    print(f"모델: {model_name}")
    print('='*60)
    
    # 학습
    cmd = [
        "python", "src/train.py",
        "--model_name", model_name,
        "--dataset_path", "data/train.json",
        "--output_dir", f"outputs/benchmark_{model_name.split('/')[-1]}",
        "--num_epochs", "1",
        "--batch_size", str(batch_size),
        "--use_lora"
    ]
    
    subprocess.run(cmd)
    
    # 평가
    subprocess.run([
        "python", "scripts/evaluate_model.py",
        "--model_path", f"outputs/benchmark_{model_name.split('/')[-1]}",
        "--eval_data", "data/eval.json",
        "--output_path", f"benchmark_{model_name.split('/')[-1]}.json"
    ])

print("\n벤치마크 완료!")
```

### 모델 자동 선택기

```python
# src/model_selector.py (새 파일)
"""
작업에 맞는 모델 자동 선택
"""

import torch

def select_model_for_task(
    task: str,
    available_memory_gb: float = None
) -> str:
    """
    작업과 하드웨어에 맞는 모델 선택
    
    Args:
        task: 'general', 'code', 'translation', 'chat' 등
        available_memory_gb: 사용 가능한 GPU 메모리 (GB)
    """
    
    # GPU 메모리 자동 감지
    if available_memory_gb is None and torch.cuda.is_available():
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # 메모리별 추천
    if available_memory_gb is None or available_memory_gb < 8:
        # CPU 또는 작은 GPU
        return "gpt2"
    
    elif available_memory_gb < 16:
        # 8-16GB GPU
        return "microsoft/phi-2"
    
    elif available_memory_gb < 24:
        # 16-24GB GPU
        if task == "code":
            return "codellama/CodeLlama-7b-hf"
        else:
            return "mistralai/Mistral-7B-v0.1"
    
    else:
        # 24GB+ GPU
        if task == "code":
            return "codellama/CodeLlama-13b-hf"
        elif task == "translation":
            return "facebook/nllb-200-3.3B"
        else:
            return "meta-llama/Llama-2-13b-hf"

# 사용
if __name__ == "__main__":
    recommended_model = select_model_for_task("general")
    print(f"추천 모델: {recommended_model}")
```

---

## 🎓 모델 추가 체크리스트

새 모델을 추가할 때 확인할 사항:

### ✅ 필수 확인
- [ ] Hugging Face에서 모델 페이지 확인
- [ ] 라이선스 확인 (상업적 사용 가능한지)
- [ ] 호환성 테스트 실행
- [ ] 테스트 학습 실행 (1 에포크)
- [ ] 추론 테스트

### ✅ 선택 확인
- [ ] 최적 LoRA 설정 찾기
- [ ] 배치 크기 조정
- [ ] `supported_models.yaml`에 추가
- [ ] README.md 업데이트

### ✅ 특수 요구사항
- [ ] `trust_remote_code` 필요 여부
- [ ] 인증 필요 여부 (Hugging Face 로그인)
- [ ] 특수 토크나이저 설정

---

## 📊 모델 비교표

### 인기 모델 비교

| 모델 | 크기 | 속도 | 품질 | 상업용 | 특이사항 |
|------|------|------|------|--------|----------|
| GPT-2 | 124M | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | 테스트용 |
| Phi-2 | 2.7B | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | 작지만 강력 |
| Mistral-7B | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 균형잡힌 |
| Llama-2-7B | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | 인기 많음 |
| Gemma-7B | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | Google 최신 |
| Qwen-7B | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 다국어 강함 |

---

## 🛠️ 고급: 완전히 새로운 아키텍처

완전히 새로운 모델 아키텍처를 추가하려면:

### 1. 커스텀 모델 클래스 정의

```python
# src/custom_architecture.py
from transformers import PreTrainedModel, PretrainedConfig

class MyCustomConfig(PretrainedConfig):
    model_type = "my_custom_model"
    
    def __init__(self, vocab_size=50000, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

class MyCustomModel(PreTrainedModel):
    config_class = MyCustomConfig
    
    def __init__(self, config):
        super().__init__(config)
        # 모델 레이어 정의
        ...
    
    def forward(self, input_ids, **kwargs):
        # 순전파
        ...

# 등록
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("my_custom_model", MyCustomConfig)
AutoModelForCausalLM.register(MyCustomConfig, MyCustomModel)
```

### 2. train.py에서 임포트

```python
# src/train.py 상단에 추가
try:
    from custom_architecture import MyCustomModel
except ImportError:
    pass
```

---

## 🎯 실전 시나리오

### 시나리오 1: "CodeLlama를 추가하고 싶어요"

```bash
# 바로 사용 가능!
python src/train.py \
    --model_name "codellama/CodeLlama-7b-hf" \
    --dataset_path "data/code_dataset.json" \
    --use_lora
```

### 시나리오 2: "회사 자체 모델을 사용하고 싶어요"

```bash
# 로컬 모델 경로 사용
python src/train.py \
    --model_name "/path/to/company/model" \
    --dataset_path "data/company_data.json"
```

### 시나리오 3: "최신 SOTA 모델을 테스트하고 싶어요"

```bash
# 1. 호환성 먼저 체크
python scripts/check_model_compatibility.py "new-model/sota-7b" --quick

# 2. 작은 데이터로 테스트
python src/train.py \
    --model_name "new-model/sota-7b" \
    --dataset_path "data/train.json" \
    --num_epochs 1

# 3. 성공하면 본격 학습
python src/train.py --config configs/train_config.yaml
```

---

## 📚 참고 자료

- **Hugging Face Models**: https://huggingface.co/models
- **지원 모델 목록**: `configs/supported_models.yaml`
- **호환성 체크**: `scripts/check_model_compatibility.py`
- **상세 가이드**: `MODEL_EXTENSION_GUIDE.md`

---

## 요약

### 대부분의 경우

```bash
# 그냥 모델 이름만 바꾸면 됩니다!
python src/train.py --model_name "원하는-모델"
```

### 특수한 경우만

- Qwen, Falcon → `trust_remote_code: true`
- Gemma → `huggingface-cli login`
- 큰 모델 → `quantization` 활성화

**50개 이상의 모델이 즉시 사용 가능합니다!** 🎉
