# 클라이언트에서 LLM 사용 가이드

이 프로젝트의 API를 통해 클라이언트에서 LLM을 사용하는 완전한 가이드입니다.

## 현재 프로젝트의 API 서버

이 프로젝트에는 **2개의 API 서버**가 있습니다:

### 1. Inference API (추론/사용) - `api_server.py`
- **용도**: 학습된 모델로 추론/질문 응답
- **포트**: 8000 (기본)
- **기능**: 텍스트 생성, 채팅

### 2. Training API (학습 관리) - `training_api.py`  
- **용도**: 학습 작업 시작/관리
- **포트**: 8001 (기본)
- **기능**: SFT/DPO 학습, 작업 모니터링

---

## 빠른 시작 (3분)

### 1단계: API 서버 시작

```bash
# 터미널 1: Inference API 시작
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --port 8000
```

서버 시작 확인:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 2단계: 클라이언트에서 사용

#### Python 클라이언트

```python
import requests

# API 서버 URL
API_URL = "http://localhost:8000"

# 1. 헬스체크
response = requests.get(f"{API_URL}/health")
print(response.json())
# {'status': 'healthy', 'model_loaded': True, ...}

# 2. 질문하기
response = requests.post(
    f"{API_URL}/chat",
    json={
        "instruction": "Python이란 무엇인가요?",
        "max_new_tokens": 200,
        "temperature": 0.7
    }
)

result = response.json()
print(result['response'])
```

#### cURL로 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# 질문하기
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Python의 장점은?",
    "max_new_tokens": 200
  }'
```

---

## 클라이언트 라이브러리 사용

### 제공되는 Python 클라이언트

이 프로젝트에는 `scripts/api_client.py`가 포함되어 있습니다!

```python
from scripts.api_client import LLMClient

# 클라이언트 초기화
client = LLMClient("http://localhost:8000")

# 헬스체크
health = client.health_check()
print(f"서버 상태: {health['status']}")

# 질문하기
result = client.chat(
    instruction="머신러닝이란?",
    max_new_tokens=200,
    temperature=0.7
)

print(result['response'])
```

---

## 실전 예제

### 예제 1: 웹 애플리케이션 (Flask)

```python
# app.py
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
LLM_API_URL = "http://localhost:8000"

@app.route('/ask', methods=['POST'])
def ask_question():
    """사용자 질문 처리"""
    data = request.json
    question = data.get('question')
    
    # LLM API 호출
    response = requests.post(
        f"{LLM_API_URL}/chat",
        json={
            "instruction": question,
            "max_new_tokens": 200
        }
    )
    
    result = response.json()
    return jsonify({
        'question': question,
        'answer': result['response']
    })

if __name__ == '__main__':
    app.run(port=5000)
```

사용:
```bash
# 서버 시작
python app.py

# 요청
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Python이란?"}'
```

### 예제 2: React 프론트엔드

```javascript
// ChatComponent.jsx
import React, { useState } from 'react';

function ChatComponent() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instruction: question,
          max_new_tokens: 200,
          temperature: 0.7
        })
      });
      
      const data = await response.json();
      setAnswer(data.response);
    } catch (error) {
      console.error('Error:', error);
    }
    
    setLoading(false);
  };

  return (
    <div>
      <input
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="질문을 입력하세요"
      />
      <button onClick={askQuestion} disabled={loading}>
        {loading ? '생성 중...' : '질문하기'}
      </button>
      {answer && <div><strong>답변:</strong> {answer}</div>}
    </div>
  );
}
```

### 예제 3: Python 채팅봇

```python
# chatbot.py
from scripts.api_client import LLMClient

def chatbot():
    """간단한 채팅봇"""
    client = LLMClient("http://localhost:8000")
    
    print("채팅봇을 시작합니다! (종료: 'quit')")
    
    while True:
        question = input("\n당신: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("채팅을 종료합니다.")
            break
        
        if not question:
            continue
        
        try:
            result = client.chat(
                instruction=question,
                max_new_tokens=200
            )
            print(f"AI: {result['response']}")
        
        except Exception as e:
            print(f"오류: {e}")

if __name__ == "__main__":
    chatbot()
```

실행:
```bash
python chatbot.py
```

### 예제 4: Streamlit 대시보드

```python
# dashboard.py
import streamlit as st
import requests

st.title("LLM 채팅 대시보드")

# API URL
API_URL = "http://localhost:8000"

# 질문 입력
question = st.text_input("질문을 입력하세요:")

# 파라미터 설정
col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("Max Tokens", 50, 500, 200)
with col2:
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

# 질문하기 버튼
if st.button("질문하기"):
    if question:
        with st.spinner("생성 중..."):
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "instruction": question,
                    "max_new_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            result = response.json()
            st.success("완료!")
            st.write("**답변:**")
            st.write(result['response'])
            st.info(f"생성 시간: {result['generation_time']:.2f}초")
```

실행:
```bash
pip install streamlit
streamlit run dashboard.py
```

---

## 다양한 언어에서 사용

### JavaScript/TypeScript

```javascript
// api-client.js
class LLMClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async chat(instruction, options = {}) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction,
        max_new_tokens: options.maxTokens || 200,
        temperature: options.temperature || 0.7
      })
    });

    const data = await response.json();
    return data.response;
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }
}

// 사용
const client = new LLMClient();
const answer = await client.chat('Python이란?');
console.log(answer);
```

### Java

```java
// LLMClient.java
import java.net.http.*;
import java.net.URI;
import org.json.*;

public class LLMClient {
    private String baseUrl;
    private HttpClient client;
    
    public LLMClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.client = HttpClient.newHttpClient();
    }
    
    public String chat(String instruction, int maxTokens) throws Exception {
        JSONObject request = new JSONObject();
        request.put("instruction", instruction);
        request.put("max_new_tokens", maxTokens);
        
        HttpRequest httpRequest = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/chat"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(request.toString()))
            .build();
        
        HttpResponse<String> response = client.send(
            httpRequest,
            HttpResponse.BodyHandlers.ofString()
        );
        
        JSONObject result = new JSONObject(response.body());
        return result.getString("response");
    }
}

// 사용
LLMClient client = new LLMClient("http://localhost:8000");
String answer = client.chat("Python이란?", 200);
System.out.println(answer);
```

### Go

```go
// llm_client.go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
)

type LLMClient struct {
    BaseURL string
}

type ChatRequest struct {
    Instruction  string  `json:"instruction"`
    MaxNewTokens int     `json:"max_new_tokens"`
    Temperature  float64 `json:"temperature"`
}

type ChatResponse struct {
    Response string `json:"response"`
}

func (c *LLMClient) Chat(instruction string) (string, error) {
    reqBody := ChatRequest{
        Instruction:  instruction,
        MaxNewTokens: 200,
        Temperature:  0.7,
    }
    
    jsonData, _ := json.Marshal(reqBody)
    
    resp, err := http.Post(
        c.BaseURL+"/chat",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()
    
    var result ChatResponse
    json.NewDecoder(resp.Body).Decode(&result)
    
    return result.Response, nil
}

// 사용
client := &LLMClient{BaseURL: "http://localhost:8000"}
answer, _ := client.Chat("Python이란?")
fmt.Println(answer)
```

---

## API 엔드포인트

### Inference API (포트 8000)

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 서버 상태 확인 |
| `/chat` | POST | 대화형 질의 (Instruction 형식) |
| `/generate` | POST | 일반 텍스트 생성 |
| `/model_info` | GET | 모델 정보 |
| `/load_model` | POST | 새 모델 로딩 |

### Training API (포트 8001)

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/train/sft` | POST | SFT 학습 시작 |
| `/train/dpo` | POST | DPO 학습 시작 |
| `/jobs` | GET | 학습 작업 목록 |
| `/jobs/{id}` | GET | 작업 상세 정보 |
| `/jobs/{id}/logs` | GET | 학습 로그 |

---

## 테스트

### 1. API 서버 테스트

```bash
# 제공된 테스트 스크립트
python scripts/test_api.py --mode all
```

### 2. 대화형 테스트

```bash
python scripts/test_api.py --mode interactive
```

### 3. 클라이언트 예제 실행

```bash
python scripts/api_client.py
```

---

## 프로덕션 배포

### Docker 컨테이너

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt requirements_api.txt ./
RUN pip install -r requirements.txt -r requirements_api.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["python", "src/api_server.py", \
     "--model_path", "models/your-model", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

빌드 및 실행:
```bash
docker build -t llm-api .
docker run -p 8000:8000 --gpus all llm-api
```

### Nginx 리버스 프록시

```nginx
# /etc/nginx/sites-available/llm-api
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

---

## 완전한 워크플로우

### 시나리오: 웹 서비스 구축

```bash
# 1. 모델 학습 (한 번만)
python src/train.py --config configs/train_config.yaml

# 2. API 서버 시작 (백그라운드)
nohup python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --port 8000 > api.log 2>&1 &

# 3. 웹 애플리케이션 시작
python your_webapp.py

# 4. 클라이언트에서 사용
# 웹, 모바일, CLI 등 어디서든 API 호출
```

---

## 요약

### ✅ 네, 완전히 가능합니다!

1. **API 서버가 이미 구축되어 있음**
   - `src/api_server.py` (추론용)
   - `src/training_api.py` (학습 관리용)

2. **클라이언트 지원**
   - Python 클라이언트 제공 (`scripts/api_client.py`)
   - 모든 언어에서 HTTP로 접근 가능

3. **사용 방법**
   ```bash
   # 서버 시작
   python src/api_server.py --model_path "your-model"
   
   # 클라이언트에서 사용
   curl http://localhost:8000/chat -d '{"instruction": "질문"}'
   ```

### 📚 관련 문서

- **API_GUIDE.md** - 완전한 API 가이드
- **QUICKSTART_API.md** - API 빠른 시작
- **scripts/api_client.py** - Python 클라이언트
- **scripts/test_api.py** - 테스트 도구

### 🎯 바로 시작하기

```bash
# 1. API 서버 시작
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model"

# 2. 브라우저에서 열기
# http://localhost:8000/docs

# 3. Python으로 사용
python scripts/api_client.py
```

**모든 클라이언트에서 REST API로 접근 가능합니다!** 🚀
