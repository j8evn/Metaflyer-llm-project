# API 서버 가이드

학습된 LLM 모델을 REST API로 서비스하는 방법을 설명합니다.

## 목차
1. [설치 및 설정](#설치-및-설정)
2. [API 서버 시작](#api-서버-시작)
3. [API 엔드포인트](#api-엔드포인트)
4. [사용 예제](#사용-예제)
5. [배포 가이드](#배포-가이드)

---

## 설치 및 설정

### 1. API 의존성 설치

```bash
# 기본 패키지
pip install -r requirements.txt

# API 서버 추가 패키지
pip install -r requirements_api.txt
```

또는 개별 설치:
```bash
pip install fastapi uvicorn[standard] pydantic aiofiles
```

### 2. 환경 변수 설정 (선택사항)

`.env` 파일 생성:
```bash
MODEL_PATH=outputs/checkpoints/final_model
LOAD_IN_8BIT=false
LOAD_IN_4BIT=false
```

---

## API 서버 시작

### 방법 1: 커맨드 라인으로 시작

```bash
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --host 0.0.0.0 \
    --port 8000
```

### 방법 2: 양자화와 함께 시작

```bash
# 8-bit 양자화
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --load_in_8bit

# 4-bit 양자화 (더 적은 메모리)
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --load_in_4bit
```

### 방법 3: 개발 모드 (자동 리로드)

```bash
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --reload
```

### 방법 4: uvicorn 직접 사용

```bash
MODEL_PATH=outputs/checkpoints/final_model \
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

서버가 시작되면:
- API 문서: http://localhost:8000/docs
- 대체 문서: http://localhost:8000/redoc
- 헬스체크: http://localhost:8000/health

---

## API 엔드포인트

### 1. 헬스체크

**GET** `/health`

서버와 모델 상태 확인

**응답 예제:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "final_model",
  "device": "cuda"
}
```

### 2. 텍스트 생성

**POST** `/generate`

일반 텍스트 생성

**요청 본문:**
```json
{
  "prompt": "Python이란 무엇인가요?",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "num_return_sequences": 1
}
```

**응답 예제:**
```json
{
  "generated_text": [
    "Python이란 무엇인가요? Python은 1991년 귀도 반 로섬이..."
  ],
  "prompt": "Python이란 무엇인가요?",
  "model_name": "final_model",
  "generation_time": 1.23
}
```

### 3. 대화형 생성

**POST** `/chat`

Instruction 형식의 대화형 생성

**요청 본문:**
```json
{
  "instruction": "Python에서 리스트를 정렬하는 방법을 알려주세요",
  "input": "",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**응답 예제:**
```json
{
  "response": "Python에서 리스트를 정렬하는 방법은...",
  "instruction": "Python에서 리스트를 정렬하는 방법을 알려주세요",
  "model_name": "final_model",
  "generation_time": 0.95
}
```

### 4. 모델 정보

**GET** `/model_info`

현재 로딩된 모델 정보

**응답 예제:**
```json
{
  "model_name": "final_model",
  "model_path": "outputs/checkpoints/final_model",
  "device": "cuda",
  "gpu_memory_allocated": "3.45 GB",
  "gpu_memory_reserved": "4.00 GB"
}
```

### 5. 모델 로딩

**POST** `/load_model`

새로운 모델 로딩

**쿼리 파라미터:**
- `model_path`: 모델 경로 (필수)
- `load_in_8bit`: 8-bit 양자화 (선택)
- `load_in_4bit`: 4-bit 양자화 (선택)

**요청 예제:**
```
POST /load_model?model_path=outputs/new_model&load_in_8bit=false
```

---

## 사용 예제

### cURL로 요청

```bash
# 헬스체크
curl http://localhost:8000/health

# 텍스트 생성
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Python이란?",
    "max_new_tokens": 100,
    "temperature": 0.7
  }'

# 대화형 생성
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "머신러닝을 설명하세요",
    "max_new_tokens": 150
  }'
```

### Python으로 요청

```python
import requests

# API 클라이언트 생성
base_url = "http://localhost:8000"

# 대화형 생성
response = requests.post(
    f"{base_url}/chat",
    json={
        "instruction": "Python이란 무엇인가요?",
        "max_new_tokens": 200,
        "temperature": 0.7
    }
)

result = response.json()
print(result['response'])
```

### 클라이언트 라이브러리 사용

```python
from scripts.api_client import LLMClient

# 클라이언트 초기화
client = LLMClient("http://localhost:8000")

# 헬스체크
health = client.health_check()
print(f"상태: {health['status']}")

# 대화
result = client.chat(
    instruction="Python의 장점을 설명하세요",
    max_new_tokens=200
)
print(result['response'])
```

### JavaScript/TypeScript로 요청

```javascript
// Fetch API
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    instruction: 'Python이란 무엇인가요?',
    max_new_tokens: 200,
    temperature: 0.7
  })
});

const data = await response.json();
console.log(data.response);
```

---

## 테스트

### 1. 기본 테스트

```bash
# 모든 엔드포인트 테스트
python scripts/test_api.py --mode all

# 헬스체크만
python scripts/test_api.py --mode health

# 대화형 테스트
python scripts/test_api.py --mode chat
```

### 2. 대화형 테스트

```bash
python scripts/test_api.py --mode interactive
```

대화형 모드에서 직접 질문하고 응답을 확인할 수 있습니다.

### 3. 클라이언트 예제 실행

```bash
# 모든 예제
python scripts/api_client.py

# 특정 예제
python scripts/api_client.py basic
python scripts/api_client.py chat
python scripts/api_client.py batch
```

---

## 배포 가이드

### 로컬 배포

```bash
# 백그라운드로 실행
nohup python src/api_server.py \
    --model_path outputs/checkpoints/final_model \
    --host 0.0.0.0 \
    --port 8000 > api.log 2>&1 &
```

### Docker 배포

`Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_api.txt

# 코드 복사
COPY src/ ./src/
COPY models/ ./models/

# 포트 노출
EXPOSE 8000

# 서버 시작
CMD ["python", "src/api_server.py", \
     "--model_path", "models/your-model", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

빌드 및 실행:
```bash
# 이미지 빌드
docker build -t llm-api .

# 컨테이너 실행
docker run -p 8000:8000 \
    --gpus all \
    -e MODEL_PATH=models/your-model \
    llm-api
```

### Kubernetes 배포

`deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: llm-api
        image: llm-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "models/your-model"
        resources:
          limits:
            nvidia.com/gpu: 1
```

### 프로덕션 설정

#### 1. Gunicorn 사용

```bash
pip install gunicorn

gunicorn src.api_server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

#### 2. Nginx 리버스 프록시

`/etc/nginx/sites-available/llm-api`:
```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

#### 3. systemd 서비스

`/etc/systemd/system/llm-api.service`:
```ini
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/llm
Environment="MODEL_PATH=/path/to/model"
ExecStart=/path/to/venv/bin/python src/api_server.py \
    --host 0.0.0.0 \
    --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

시작:
```bash
sudo systemctl start llm-api
sudo systemctl enable llm-api
sudo systemctl status llm-api
```

---

## 성능 최적화

### 1. 배치 처리

여러 요청을 배치로 처리하여 효율성 향상

### 2. 캐싱

자주 사용되는 응답 캐싱:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate(prompt: str):
    return model_manager.generate(prompt)
```

### 3. 양자화

```bash
# 4-bit 양자화로 메모리 75% 절약
python src/api_server.py \
    --model_path outputs/model \
    --load_in_4bit
```

### 4. GPU 최적화

```python
# torch.compile 사용 (PyTorch 2.0+)
model = torch.compile(model)
```

---

## 모니터링

### 로그 확인

```bash
# 실시간 로그
tail -f api.log

# 에러 로그만
grep ERROR api.log
```

### 메트릭

API는 자동으로 다음 정보를 로깅:
- 요청 시간
- 생성 시간
- GPU 메모리 사용량

---

## 보안

### 1. API 키 인증 추가

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/generate", dependencies=[Depends(verify_api_key)])
async def generate_text(request: GenerateRequest):
    ...
```

### 2. Rate Limiting

```bash
pip install slowapi

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_text(request: Request, data: GenerateRequest):
    ...
```

---

## 트러블슈팅

### 문제 1: 모델 로딩 실패

```bash
# 모델 경로 확인
ls outputs/checkpoints/final_model

# 로그 확인
python src/api_server.py --model_path outputs/model 2>&1 | tee api.log
```

### 문제 2: GPU 메모리 부족

```bash
# 4-bit 양자화 사용
python src/api_server.py --model_path outputs/model --load_in_4bit
```

### 문제 3: 응답 속도 느림

- 배치 크기 증가
- 양자화 사용
- GPU 사용 확인

---

더 자세한 내용은 FastAPI 문서를 참조하세요: https://fastapi.tiangolo.com/

