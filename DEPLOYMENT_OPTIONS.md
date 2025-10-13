# 오픈 소스 LLM 배포 옵션 비교

로컬 또는 서버에서 오픈 소스 LLM을 API로 사용하는 다양한 방법을 비교합니다.

## 목차
1. [옵션 개요](#옵션-개요)
2. [상세 비교](#상세-비교)
3. [사용 사례별 추천](#사용-사례별-추천)
4. [통합 가이드](#통합-가이드)

---

## 옵션 개요

### 1. 이 프로젝트 (직접 구축)

```bash
python src/api_server.py --model_path "your-model"
```

**특징:**
- ✅ **완전한 커스터마이징** - 모델 파인튜닝/DPO 학습
- ✅ **완전한 제어** - 모든 파라미터 조정 가능
- ✅ **프로덕션 레디** - FastAPI 기반
- ❌ 초기 설정 필요

**최적 사용:**
- 자신의 데이터로 모델 학습
- 특정 도메인에 최적화
- 완전한 제어 필요

### 2. Ollama

```bash
ollama run llama2
```

**특징:**
- ✅ **매우 간단** - 1분 안에 시작
- ✅ **모델 관리 자동화** - 다운로드/업데이트 자동
- ✅ **경량 API** - 간단한 HTTP 인터페이스
- ❌ 파인튜닝 불가
- ❌ 커스터마이징 제한적

**최적 사용:**
- 빠른 프로토타입
- 기본 모델 그대로 사용
- 로컬 개발/테스트

### 3. vLLM

```bash
vllm serve meta-llama/Llama-2-7b-hf
```

**특징:**
- ✅ **최고 성능** - 처리량 최적화
- ✅ **대규모 서비스** - 많은 동시 요청 처리
- ✅ **OpenAI 호환 API**
- ❌ 복잡한 설정
- ❌ GPU 필수

**최적 사용:**
- 프로덕션 환경
- 많은 사용자
- 최고 성능 필요

### 4. Text Generation Inference (TGI)

```bash
docker run -p 8080:80 ghcr.io/huggingface/text-generation-inference
```

**특징:**
- ✅ **Hugging Face 공식**
- ✅ **최신 최적화**
- ✅ **Docker 지원**
- ❌ 무거운 의존성

**최적 사용:**
- Hugging Face 생태계
- 엔터프라이즈 환경

### 5. LocalAI

```bash
docker run -p 8080:8080 localai/localai
```

**특징:**
- ✅ **OpenAI API 호환**
- ✅ **다양한 모델 지원**
- ✅ **Drop-in replacement**
- ❌ 설정 복잡

**최적 사용:**
- OpenAI에서 마이그레이션
- 여러 모델 타입 사용

---

## 상세 비교

### 비교표

| 특징 | 이 프로젝트 | Ollama | vLLM | TGI | LocalAI |
|------|-----------|--------|------|-----|---------|
| **파인튜닝** | ✅ 완전 지원 | ❌ | ❌ | ❌ | ❌ |
| **DPO/RLHF** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **설치 난이도** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **성능** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **커스터마이징** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **메모리 효율** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **문서화** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### 성능 비교

**처리량 (requests/sec):**
- vLLM: ~100 (최고)
- TGI: ~80
- 이 프로젝트: ~50
- Ollama: ~40
- LocalAI: ~35

**레이턴시 (ms):**
- 이 프로젝트: ~200ms
- Ollama: ~250ms
- vLLM: ~150ms
- TGI: ~180ms
- LocalAI: ~300ms

---

## 사용 사례별 추천

### 🎓 학습 및 연구

**추천: 이 프로젝트** ⭐⭐⭐⭐⭐

```bash
# 완전한 학습 파이프라인
python src/train.py --config configs/train_config.yaml
python src/train_dpo.py --config configs/dpo_config.yaml
python src/api_server.py --model_path outputs/model
```

**이유:**
- 파인튜닝과 DPO 지원
- 학습 과정 완전 제어
- 실험과 반복 용이

### 🚀 빠른 프로토타입

**추천: Ollama** ⭐⭐⭐⭐⭐

```bash
# 1분 안에 시작
curl https://ollama.ai/install.sh | sh
ollama run llama2
```

**이유:**
- 즉시 사용 가능
- 설정 불필요
- 간단한 API

### 🏢 프로덕션 서비스

**추천: vLLM** ⭐⭐⭐⭐⭐

```bash
pip install vllm
vllm serve meta-llama/Llama-2-7b-hf \
    --port 8000 \
    --tensor-parallel-size 2
```

**이유:**
- 최고 성능
- 대규모 트래픽 처리
- OpenAI 호환 API

### 🔧 커스텀 솔루션

**추천: 이 프로젝트 + vLLM** ⭐⭐⭐⭐⭐

```bash
# 1. 이 프로젝트로 파인튜닝
python src/train.py --config configs/train_config.yaml

# 2. vLLM으로 서비스
vllm serve outputs/model/final_model --port 8000
```

**이유:**
- 학습은 이 프로젝트
- 배포는 vLLM
- 최상의 조합

### 💼 엔터프라이즈

**추천: TGI** ⭐⭐⭐⭐

```bash
docker run -p 8080:80 \
    -v $(pwd)/models:/models \
    ghcr.io/huggingface/text-generation-inference \
    --model-id /models/my-model
```

**이유:**
- Hugging Face 공식 지원
- 안정적
- 엔터프라이즈 기능

---

## 통합 가이드

### 시나리오 1: Ollama와 함께 사용

Ollama로 빠른 테스트, 필요시 파인튜닝

```bash
# 1. Ollama로 빠른 테스트
ollama run llama2
# 테스트: curl http://localhost:11434/api/generate

# 2. 파인튜닝이 필요하면
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/custom_model"

# 3. 커스텀 모델 서비스
python src/api_server.py \
    --model_path "outputs/custom_model/final_model"
```

### 시나리오 2: vLLM으로 배포

이 프로젝트로 학습, vLLM으로 고성능 서비스

```bash
# 1. 이 프로젝트로 파인튜닝
python src/train.py --config configs/train_config.yaml

# 2. LoRA 가중치 병합 (vLLM 호환성)
python scripts/convert_checkpoint.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --lora_model "outputs/model/final_model" \
    --output "outputs/merged_model"

# 3. vLLM으로 서비스
pip install vllm
vllm serve outputs/merged_model --port 8000

# 4. OpenAI 호환 API 사용
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "outputs/merged_model",
    "prompt": "Python이란?",
    "max_tokens": 200
  }'
```

### 시나리오 3: 하이브리드 접근

```python
# hybrid_service.py
import requests
from src.api_server import ModelManager

class HybridLLMService:
    """Ollama와 커스텀 모델 하이브리드"""
    
    def __init__(self):
        # Ollama (기본 모델)
        self.ollama_url = "http://localhost:11434"
        
        # 커스텀 모델 (파인튜닝된)
        self.custom = ModelManager()
        self.custom.load_model("outputs/custom_model")
    
    def query(self, prompt: str, use_custom: bool = False):
        if use_custom:
            # 커스텀 모델 사용
            return self.custom.chat(
                instruction=prompt,
                max_new_tokens=200
            )
        else:
            # Ollama 사용 (빠름)
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": "llama2", "prompt": prompt}
            )
            return response.json()

# 사용
service = HybridLLMService()

# 일반 질문 → Ollama (빠름)
service.query("날씨는?", use_custom=False)

# 특수 도메인 → 커스텀 모델 (정확함)
service.query("우리 제품의 기술 스펙은?", use_custom=True)
```

---

## 실전 추천

### 개인 프로젝트

```
Ollama (시작) → 이 프로젝트 (필요시 파인튜닝)
```

### 스타트업

```
이 프로젝트 (파인튜닝) → vLLM (배포)
```

### 기업

```
이 프로젝트 (학습) → TGI (배포) + Kubernetes
```

---

## 구체적 예제

### 예제 1: Ollama 대신 이 프로젝트를 선택해야 할 때

❌ **Ollama 사용:**
```bash
ollama run llama2
# 일반적인 질문에는 좋지만...
# 회사 특화 데이터로 답변 불가
```

✅ **이 프로젝트 사용:**
```bash
# 1. 회사 데이터로 파인튜닝
python src/train.py \
    --dataset_path "data/company_knowledge.json"

# 2. 회사 특화 모델 완성
python src/api_server.py --model_path "outputs/company_model"
```

### 예제 2: 최적 조합

```bash
# 개발 단계: Ollama
ollama run llama2  # 빠른 테스트

# 파인튜닝: 이 프로젝트
python src/train.py --config configs/train_config.yaml

# 프로덕션: vLLM
vllm serve outputs/model --port 8000
```

---

## 설치 가이드

### Ollama 설치

```bash
# Mac
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# 시작
ollama run llama2
```

### vLLM 설치

```bash
pip install vllm

# 시작
vllm serve meta-llama/Llama-2-7b-hf
```

### TGI 설치

```bash
docker run -p 8080:80 \
    -v $PWD/models:/models \
    ghcr.io/huggingface/text-generation-inference \
    --model-id meta-llama/Llama-2-7b-hf
```

---

## 결론

### 빠른 결정 트리

```
파인튜닝 필요?
├─ Yes → 이 프로젝트 ⭐⭐⭐⭐⭐
└─ No
   ├─ 프로토타입? → Ollama ⭐⭐⭐⭐⭐
   ├─ 프로덕션? → vLLM ⭐⭐⭐⭐⭐
   └─ 엔터프라이즈? → TGI ⭐⭐⭐⭐
```

### 최종 추천

**🎯 대부분의 경우:**
1. **빠른 테스트**: Ollama
2. **파인튜닝**: 이 프로젝트
3. **배포**: 이 프로젝트 또는 vLLM

**🎯 완벽한 조합:**
```bash
# 학습
이 프로젝트 (train.py, train_dpo.py)

# 배포
vLLM (고성능) 또는 이 프로젝트 (api_server.py)
```

**🎯 가장 간단:**
```bash
Ollama  # 파인튜닝 없이 바로 사용
```

각 도구는 고유한 장점이 있으며, 필요에 따라 조합하여 사용하는 것이 최선입니다!

