# 지원 모델 확장 가이드

이 프로젝트에 새로운 LLM 모델을 추가하고 지원하는 방법을 설명합니다.

## 목차
1. [현재 지원 모델](#현재-지원-모델)
2. [새 Hugging Face 모델 추가](#새-hugging-face-모델-추가)
3. [커스텀 아키텍처 지원](#커스텀-아키텍처-지원)
4. [다른 형식 모델 통합](#다른-형식-모델-통합)
5. [멀티모달 모델 지원](#멀티모달-모델-지원)

---

## 현재 지원 모델

### 즉시 사용 가능한 모델들

이 프로젝트는 Hugging Face Transformers 기반이므로, **대부분의 Causal LM 모델**을 즉시 지원합니다:

#### 텍스트 생성 모델
- ✅ **Llama 2/3** (Meta)
- ✅ **Mistral/Mixtral** (Mistral AI)
- ✅ **Falcon** (TII)
- ✅ **GPT-2/GPT-J/GPT-Neo** (EleutherAI)
- ✅ **BLOOM** (BigScience)
- ✅ **Qwen** (Alibaba)
- ✅ **Yi** (01.AI)
- ✅ **Gemma** (Google)
- ✅ **Phi** (Microsoft)
- ✅ **StableLM** (Stability AI)

#### 사용 방법

모델 이름만 변경하면 됩니다:

```bash
# Llama 2
python src/train.py --model_name "meta-llama/Llama-2-7b-hf"

# Mistral
python src/train.py --model_name "mistralai/Mistral-7B-v0.1"

# Qwen
python src/train.py --model_name "Qwen/Qwen-7B"

# Gemma
python src/train.py --model_name "google/gemma-7b"
```

---

## 새 Hugging Face 모델 추가

### 방법 1: 모델 이름으로 직접 사용

대부분의 경우 **추가 작업 없이** 모델 이름만으로 사용 가능합니다.

```bash
# 1. 모델 검색
# https://huggingface.co/models 에서 검색

# 2. 모델 ID 복사
# 예: "upstage/SOLAR-10.7B-v1.0"

# 3. 바로 사용
python src/train.py \
    --model_name "upstage/SOLAR-10.7B-v1.0" \
    --dataset_path "data/train.json" \
    --use_lora
```

### 방법 2: 모델 설정 파일에 추가

여러 모델을 관리하려면 설정 파일을 사용:

```yaml
# configs/models.yaml
models:
  llama2-7b:
    name: "meta-llama/Llama-2-7b-hf"
    context_length: 4096
    recommended_batch_size: 4
    
  mistral-7b:
    name: "mistralai/Mistral-7B-v0.1"
    context_length: 8192
    recommended_batch_size: 4
    
  qwen-7b:
    name: "Qwen/Qwen-7B"
    context_length: 8192
    recommended_batch_size: 4
    trust_remote_code: true  # Qwen은 trust_remote_code 필요
    
  gemma-7b:
    name: "google/gemma-7b"
    context_length: 8192
    recommended_batch_size: 4
```

### 방법 3: 모델별 LoRA 타겟 커스터마이징

모델마다 최적의 LoRA 타겟 모듈이 다를 수 있습니다.

```python
# src/model_configs.py (새 파일 생성)
"""
모델별 최적 설정
"""

MODEL_CONFIGS = {
    # Llama 계열
    "llama": {
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_r": 16,
        "lora_alpha": 32
    },
    
    # Mistral 계열
    "mistral": {
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_r": 16,
        "lora_alpha": 32
    },
    
    # GPT-2 계열
    "gpt2": {
        "target_modules": [
            "c_attn", "c_proj", "c_fc"
        ],
        "lora_r": 8,
        "lora_alpha": 16
    },
    
    # Qwen 계열
    "qwen": {
        "target_modules": [
            "c_attn", "c_proj", "w1", "w2"
        ],
        "lora_r": 16,
        "lora_alpha": 32
    },
    
    # Gemma 계열
    "gemma": {
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_r": 16,
        "lora_alpha": 32
    }
}

def get_model_config(model_name: str) -> dict:
    """모델 이름에서 설정 가져오기"""
    model_name_lower = model_name.lower()
    
    for key, config in MODEL_CONFIGS.items():
        if key in model_name_lower:
            return config
    
    # 기본 설정 (Llama 스타일)
    return MODEL_CONFIGS["llama"]
```

### 방법 4: model_utils.py 확장

모델별 특수 처리가 필요한 경우:

```python
# src/model_utils.py에 추가

def load_model(self, model_name: str) -> PreTrainedModel:
    """모델 로딩 (확장 버전)"""
    logger.info(f"모델 로딩: {model_name}")
    
    # 모델별 특수 설정
    model_kwargs = self._get_model_kwargs(model_name)
    
    # 모델 로딩
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # 모델별 후처리
    model = self._post_process_model(model, model_name)
    
    return model

def _get_model_kwargs(self, model_name: str) -> dict:
    """모델별 로딩 인자"""
    kwargs = {
        'pretrained_model_name_or_path': model_name,
        'device_map': 'auto' if self.device == 'cuda' else None,
    }
    
    # Qwen: trust_remote_code 필요
    if 'qwen' in model_name.lower():
        kwargs['trust_remote_code'] = True
    
    # Falcon: trust_remote_code 필요
    if 'falcon' in model_name.lower():
        kwargs['trust_remote_code'] = True
    
    # Gemma: torch_dtype 설정
    if 'gemma' in model_name.lower():
        kwargs['torch_dtype'] = torch.bfloat16
    
    # 양자화 설정 추가
    quantization_config = self.get_quantization_config()
    if quantization_config:
        kwargs['quantization_config'] = quantization_config
    
    return kwargs

def _post_process_model(self, model, model_name: str):
    """모델 후처리"""
    # 특정 모델의 특수 처리
    if 'mpt' in model_name.lower():
        # MPT 모델 특수 처리
        model.config.attn_config['attn_impl'] = 'torch'
    
    return model
```

---

## 커스텀 아키텍처 지원

### 완전히 새로운 모델 아키텍처 추가

```python
# src/custom_models.py (새 파일)
"""
커스텀 모델 아키텍처
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class CustomModelConfig(PretrainedConfig):
    """커스텀 모델 설정"""
    model_type = "custom_model"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size


class CustomModel(PreTrainedModel):
    """커스텀 모델 구현"""
    config_class = CustomModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # 모델 레이어 정의
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        
        self.layers = nn.ModuleList([
            CustomDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # 순전파 로직
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits}


# 모델 등록
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("custom_model", CustomModelConfig)
AutoModelForCausalLM.register(CustomModelConfig, CustomModel)
```

사용:

```python
# 커스텀 모델 임포트
from src.custom_models import CustomModel, CustomModelConfig

# 학습
python src/train.py --model_name "path/to/custom_model"
```

---

## 다른 형식 모델 통합

### GGUF 모델 지원

```python
# src/gguf_loader.py (새 파일)
"""
GGUF 형식 모델 로더
"""

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python 설치 필요: pip install llama-cpp-python")

class GGUFModelWrapper:
    """GGUF 모델 래퍼"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1  # 모든 레이어를 GPU로
        )
    
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs):
        """텍스트 생성"""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
        )
        
        return output['choices'][0]['text']

# 사용
from src.gguf_loader import GGUFModelWrapper

model = GGUFModelWrapper("models/llama-2-7b.Q4_K_M.gguf")
response = model.generate("Python이란?")
```

### ONNX 모델 지원

```python
# src/onnx_loader.py (새 파일)
"""
ONNX 형식 모델 로더
"""

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime 설치 필요: pip install onnxruntime-gpu")

class ONNXModelWrapper:
    """ONNX 모델 래퍼"""
    
    def __init__(self, model_path: str):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
    
    def generate(self, input_ids, attention_mask=None):
        """추론 실행"""
        inputs = {
            'input_ids': input_ids.numpy(),
        }
        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask.numpy()
        
        outputs = self.session.run(None, inputs)
        return outputs[0]
```

---

## 멀티모달 모델 지원

### Vision-Language 모델 (LLaVA, BLIP 등)

```python
# src/multimodal_utils.py (새 파일)
"""
멀티모달 모델 지원
"""

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BlipForConditionalGeneration
)
from PIL import Image

class MultiModalModel:
    """멀티모달 모델 래퍼"""
    
    def __init__(self, model_name: str, model_type: str = "llava"):
        self.model_type = model_type
        
        if model_type == "llava":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
        
        elif model_type == "blip":
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
    
    def generate_from_image(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 256
    ):
        """이미지와 텍스트로부터 생성"""
        image = Image.open(image_path)
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)

# 사용
model = MultiModalModel("llava-hf/llava-1.5-7b-hf", model_type="llava")
response = model.generate_from_image(
    "image.jpg",
    "이 이미지를 설명해주세요"
)
```

---

## 실전 예제

### 예제 1: Qwen 모델 추가

```bash
# 1. 기본 사용
python src/train.py \
    --model_name "Qwen/Qwen-7B-Chat" \
    --dataset_path "data/train.json" \
    --use_lora

# 2. trust_remote_code 설정 필요 시
# configs/train_config.yaml
model:
  name: "Qwen/Qwen-7B-Chat"
  trust_remote_code: true
```

### 예제 2: Gemma 모델 추가

```bash
# Gemma는 별도 인증 필요
huggingface-cli login

# 학습
python src/train.py \
    --model_name "google/gemma-7b" \
    --dataset_path "data/train.json" \
    --use_lora
```

### 예제 3: 로컬 모델 사용

```bash
# 1. 모델 다운로드
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf models/llama2

# 2. 로컬 경로로 학습
python src/train.py \
    --model_name "models/llama2" \
    --dataset_path "data/train.json"
```

### 예제 4: 여러 모델 벤치마크

```python
# scripts/benchmark_models.py (새 파일)
"""
여러 모델 성능 비교
"""

import subprocess

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b",
    "Qwen/Qwen-7B"
]

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"모델: {model}")
    print('='*60)
    
    # 학습
    subprocess.run([
        "python", "src/train.py",
        "--model_name", model,
        "--dataset_path", "data/train.json",
        "--output_dir", f"outputs/{model.split('/')[-1]}",
        "--num_epochs", "1",
        "--use_lora"
    ])
    
    # 평가
    subprocess.run([
        "python", "scripts/evaluate_model.py",
        "--model_path", f"outputs/{model.split('/')[-1]}",
        "--eval_data", "data/eval.json"
    ])
```

---

## 모델 호환성 확인

### 자동 호환성 체크 스크립트

```python
# scripts/check_model_compatibility.py (새 파일)
"""
모델 호환성 확인
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def check_model(model_name: str):
    """모델 호환성 체크"""
    print(f"모델 확인: {model_name}")
    print("-" * 60)
    
    try:
        # 토크나이저 체크
        print("1. 토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   ✓ 성공 (vocab size: {len(tokenizer)})")
        
        # 모델 체크
        print("2. 모델 로딩...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype="auto"
        )
        print(f"   ✓ 성공")
        
        # 파라미터 수
        num_params = sum(p.numel() for p in model.parameters())
        print(f"3. 파라미터 수: {num_params / 1e9:.2f}B")
        
        # 테스트 생성
        print("4. 테스트 생성...")
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        text = tokenizer.decode(outputs[0])
        print(f"   ✓ 성공: {text[:50]}...")
        
        print("\n✅ 이 모델은 호환됩니다!")
        return True
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python check_model_compatibility.py <model-name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    check_model(model_name)
```

사용:

```bash
python scripts/check_model_compatibility.py "mistralai/Mistral-7B-v0.1"
```

---

## 모델 레지스트리

### 지원 모델 목록 관리

```python
# src/model_registry.py (새 파일)
"""
지원 모델 레지스트리
"""

SUPPORTED_MODELS = {
    # Meta AI
    "llama-2-7b": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "size": "7B",
        "context_length": 4096,
        "license": "Llama 2 License",
        "verified": True
    },
    "llama-2-13b": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "size": "13B",
        "context_length": 4096,
        "license": "Llama 2 License",
        "verified": True
    },
    
    # Mistral AI
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "size": "7B",
        "context_length": 8192,
        "license": "Apache 2.0",
        "verified": True
    },
    
    # Google
    "gemma-7b": {
        "hf_id": "google/gemma-7b",
        "size": "7B",
        "context_length": 8192,
        "license": "Gemma License",
        "auth_required": True,
        "verified": True
    },
    
    # Alibaba
    "qwen-7b": {
        "hf_id": "Qwen/Qwen-7B",
        "size": "7B",
        "context_length": 8192,
        "license": "Tongyi Qianwen License",
        "trust_remote_code": True,
        "verified": True
    }
}

def list_models():
    """지원 모델 목록"""
    print("지원 모델 목록:")
    print("=" * 80)
    
    for name, info in SUPPORTED_MODELS.items():
        status = "✓" if info.get("verified") else "?"
        auth = " [인증 필요]" if info.get("auth_required") else ""
        trust = " [trust_remote_code]" if info.get("trust_remote_code") else ""
        
        print(f"{status} {name:20s} | {info['size']:4s} | {info['license']}{auth}{trust}")

def get_model_info(name: str):
    """모델 정보 가져오기"""
    return SUPPORTED_MODELS.get(name)

if __name__ == "__main__":
    list_models()
```

---

## 요약

### 새 모델 추가 체크리스트

1. ✅ **Hugging Face에서 모델 확인**
   - https://huggingface.co/models

2. ✅ **호환성 테스트**
   ```bash
   python scripts/check_model_compatibility.py "model-name"
   ```

3. ✅ **기본 학습 테스트**
   ```bash
   python src/train.py \
       --model_name "model-name" \
       --dataset_path "data/train.json" \
       --num_epochs 1
   ```

4. ✅ **설정 파일에 추가** (선택)
   - configs/models.yaml
   - src/model_registry.py

5. ✅ **문서 업데이트**
   - README.md에 지원 모델 목록 추가

### 대부분의 경우

**아무것도 수정할 필요 없이** 모델 이름만 변경하면 됩니다:

```bash
python src/train.py --model_name "새로운-모델-이름"
```

이게 전부입니다! 🚀

