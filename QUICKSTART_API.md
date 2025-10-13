# API 서버 빠른 시작

학습된 모델을 REST API로 5분 안에 서비스하는 방법입니다.

## 1단계: API 의존성 설치 (1분)

```bash
pip install -r requirements_api.txt
```

또는:
```bash
pip install fastapi uvicorn[standard]
```

## 2단계: API 서버 시작 (1분)

### 방법 A: 스크립트 사용 (가장 쉬움)

```bash
./scripts/start_api.sh
```

### 방법 B: 직접 실행

```bash
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --port 8000
```

서버가 시작되면:
- **API 문서**: http://localhost:8000/docs
- **헬스체크**: http://localhost:8000/health

## 3단계: API 테스트 (1분)

### 브라우저에서

http://localhost:8000/docs 에서 직접 테스트!

### cURL로

```bash
# 헬스체크
curl http://localhost:8000/health

# 대화 생성
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Python이란 무엇인가요?",
    "max_new_tokens": 200
  }'
```

### Python으로

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "instruction": "Python의 장점을 설명하세요",
        "max_new_tokens": 200
    }
)

print(response.json()['response'])
```

## 4단계: 클라이언트 사용 (1분)

```python
from scripts.api_client import LLMClient

client = LLMClient("http://localhost:8000")

# 대화
result = client.chat(
    instruction="머신러닝이란?",
    max_new_tokens=200
)

print(result['response'])
```

## 주요 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 서버 상태 확인 |
| `/chat` | POST | 대화형 생성 (권장) |
| `/generate` | POST | 일반 텍스트 생성 |
| `/model_info` | GET | 모델 정보 |

## 옵션: 양자화로 시작

메모리가 부족하다면:

```bash
# 8-bit 양자화
python src/api_server.py \
    --model_path "outputs/model" \
    --load_in_8bit

# 4-bit 양자화 (더 적은 메모리)
python src/api_server.py \
    --model_path "outputs/model" \
    --load_in_4bit
```

## 자동 테스트

```bash
# 전체 테스트
python scripts/test_api.py --mode all

# 대화형 테스트
python scripts/test_api.py --mode interactive
```

## 다음 단계

- **프로덕션 배포**: `API_GUIDE.md` 참조
- **Docker 배포**: `API_GUIDE.md`의 Docker 섹션 참조
- **보안 설정**: API 키 인증 추가

더 자세한 내용은 `API_GUIDE.md`를 참조하세요!
