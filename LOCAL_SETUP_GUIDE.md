# 로컬에서 LLM 실행 가이드

처음부터 끝까지 로컬 컴퓨터에서 LLM을 학습하고 실행하는 완전한 가이드입니다.

## 목차
1. [환경 설정](#1-환경-설정)
2. [모델 학습](#2-모델-학습)
3. [모델 추론](#3-모델-추론)
4. [API 서버 실행](#4-api-서버-실행-선택)
5. [트러블슈팅](#5-트러블슈팅)

---

## 1. 환경 설정

### 1단계: 필수 요구사항 확인

```bash
# Python 버전 확인 (3.8 이상)
python --version

# pip 확인
pip --version

# Git 확인 (선택)
git --version
```

### 2단계: 프로젝트 디렉토리 이동

```bash
cd /Users/jerry/metaflyer/llm
```

### 3단계: 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate  # Mac/Linux

# 성공하면 프롬프트에 (venv) 표시됨
```

### 4단계: 의존성 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# 설치 확인
pip list | grep torch
pip list | grep transformers
```

**설치 시간:** 5-10분 (인터넷 속도에 따라)

---

## 2. 모델 학습

### 옵션 A: 샘플 데이터로 빠른 테스트 (권장)

```bash
# 1. GPT-2로 빠른 테스트 (5-10분)
python src/train.py \
    --model_name "gpt2" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/gpt2_test" \
    --num_epochs 1 \
    --batch_size 2
```

터미널 출력 예시:
```
모델 로딩: gpt2
데이터 로딩: data/train.json
학습 시작...
Epoch 1/1: 100%|████████| 3/3 [00:15<00:00]
Loss: 2.345
학습 완료!
모델 저장: outputs/gpt2_test/final_model
```

### 옵션 B: 실전 파인튜닝

#### Llama-2 7B 모델 (권장)

```bash
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/llama2_finetuned" \
    --use_lora \
    --num_epochs 3 \
    --batch_size 4
```

**필요 사항:**
- GPU: 16GB 이상 (LoRA 사용 시)
- 시간: 1-3시간 (데이터 크기에 따라)

#### 설정 파일 사용 (추천)

```bash
# 1. 설정 파일 수정
nano configs/train_config.yaml

# 2. 학습 실행
python src/train.py --config configs/train_config.yaml
```

`configs/train_config.yaml` 예시:
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"

data:
  train_path: "data/train.json"
  max_length: 512

lora:
  use_lora: true
  r: 16

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
```

### DPO 강화학습 (선택)

SFT 학습 완료 후:

```bash
python src/train_dpo.py \
    --model_name "outputs/llama2_finetuned/final_model" \
    --dataset_path "data/preference_train.json" \
    --output_dir "outputs/llama2_dpo" \
    --beta 0.1 \
    --num_epochs 1
```

---

## 3. 모델 추론

학습이 완료되면 모델을 바로 사용할 수 있습니다!

### 방법 1: 대화형 모드 (가장 쉬움)

```bash
python src/inference.py \
    --model_path "outputs/gpt2_test/final_model"
```

대화형 인터페이스:
```
질문을 입력하세요: Python이란 무엇인가요?

생성 중...

응답:
Python은 1991년 귀도 반 로섬이 개발한 고수준 프로그래밍 언어입니다...

질문을 입력하세요: (계속 질문 가능)
```

종료: `quit`, `exit`, 또는 `q` 입력

### 방법 2: 단일 질문

```bash
python src/inference.py \
    --model_path "outputs/gpt2_test/final_model" \
    --instruction "Python의 장점을 설명하세요" \
    --max_new_tokens 200
```

### 방법 3: Python 코드로

```python
# test_inference.py
from src.inference import InferenceEngine

# 엔진 초기화
engine = InferenceEngine(
    model_path="outputs/gpt2_test/final_model"
)

# 질문
response = engine.chat(
    instruction="머신러닝이란 무엇인가요?",
    max_new_tokens=200
)

print(response)
```

실행:
```bash
python test_inference.py
```

---

## 4. API 서버 실행 (선택)

웹 애플리케이션이나 원격 접속이 필요한 경우:

### Inference API (추론 서버)

```bash
# 터미널 1: 서버 시작
python src/api_server.py \
    --model_path "outputs/gpt2_test/final_model" \
    --port 8000
```

서버 시작 확인:
```
INFO: Started server process
INFO: Uvicorn running on http://0.0.0.0:8000
```

브라우저에서 열기:
- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health

### API로 질문하기

#### cURL로:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Python이란?",
    "max_new_tokens": 200
  }'
```

#### Python으로:

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "instruction": "Python의 장점은?",
        "max_new_tokens": 200
    }
)

result = response.json()
print(result['response'])
```

---

## 5. 트러블슈팅

### 문제 1: GPU 메모리 부족

```bash
# 해결책 1: 4-bit 양자화
python src/inference.py \
    --model_path "outputs/model" \
    --load_in_4bit

# 해결책 2: CPU 사용
python src/inference.py \
    --model_path "outputs/model"
# (자동으로 CPU 감지)
```

### 문제 2: 모델 다운로드 실패

```bash
# Hugging Face 로그인 필요
pip install huggingface_hub
huggingface-cli login

# 토큰 입력 후 재시도
```

### 문제 3: 패키지 충돌

```bash
# 가상환경 삭제 및 재생성
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 문제 4: 학습 속도 느림

```bash
# GPU 사용 확인
python -c "import torch; print(torch.cuda.is_available())"

# True여야 함. False면 CPU 사용 중
```

---

## 완전한 실습 예제

### 처음부터 끝까지 (15분 테스트)

```bash
# 1. 환경 설정
cd /Users/jerry/metaflyer/llm
source venv/bin/activate  # 이미 설치했다면

# 2. 빠른 학습 (GPT-2, 1 에포크)
python src/train.py \
    --model_name "gpt2" \
    --dataset_path "data/train.json" \
    --output_dir "outputs/test_model" \
    --num_epochs 1 \
    --batch_size 2

# 3. 추론 테스트
python src/inference.py \
    --model_path "outputs/test_model/final_model" \
    --instruction "Python이란 무엇인가요?"

# 4. 완료!
```

### 실전 프로젝트 (1-2일)

```bash
# Day 1: SFT 학습
python src/train.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "data/my_dataset.json" \
    --output_dir "outputs/my_llama2" \
    --use_lora \
    --num_epochs 3

# Day 2: DPO 학습
python src/train_dpo.py \
    --model_name "outputs/my_llama2/final_model" \
    --dataset_path "data/preference_data.json" \
    --output_dir "outputs/my_llama2_dpo" \
    --num_epochs 1

# 배포: API 서버
python src/api_server.py \
    --model_path "outputs/my_llama2_dpo/final_model" \
    --port 8000
```

---

## 시스템 요구사항

### 최소 사양 (GPT-2 테스트용)

- CPU: 4코어 이상
- RAM: 8GB
- 디스크: 10GB
- GPU: 선택사항

### 권장 사양 (Llama-2 7B)

- CPU: 8코어 이상
- RAM: 32GB
- 디스크: 50GB
- GPU: 16GB VRAM (RTX 3090, A100 등)

### 대용량 모델 (13B+)

- RAM: 64GB+
- GPU: 40GB+ VRAM (A100)
- 또는 여러 GPU

---

## 다음 단계

1. ✅ 환경 설정 완료
2. ✅ 샘플로 테스트
3. ✅ 자신의 데이터 준비
4. ✅ 실전 학습
5. ✅ 모델 사용

더 자세한 내용:
- **학습**: `README.md`, `EXAMPLES.md`
- **DPO**: `DPO_GUIDE.md`
- **API**: `API_GUIDE.md`
- **빠른 시작**: `QUICKSTART.md`

