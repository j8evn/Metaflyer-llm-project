# vLLM 클라이언트 사용 가이드

vLLM으로 서비스되는 LLM에 접근하는 다양한 방법을 설명합니다.

## 목차
1. [vLLM 서버 시작](#vllm-서버-시작)
2. [OpenAI Python 라이브러리](#1-openai-python-라이브러리-권장)
3. [HTTP 요청](#2-http-요청-curlrequests)
4. [LangChain](#3-langchain)
5. [기타 클라이언트](#4-기타-클라이언트)

---

## vLLM 서버 시작

먼저 vLLM 서버를 시작합니다:

### 설치

```bash
pip install vllm
```

### 기본 서버 시작

```bash
# 기본 모델
vllm serve meta-llama/Llama-2-7b-hf --port 8000

# 파인튜닝한 모델
vllm serve outputs/my_model/final_model --port 8000

# 여러 GPU 사용
vllm serve meta-llama/Llama-2-7b-hf \
    --port 8000 \
    --tensor-parallel-size 2
```

서버가 시작되면:
- **OpenAI 호환 API**: http://localhost:8000/v1
- **문서**: http://localhost:8000/docs

---

## 1. OpenAI Python 라이브러리 (권장) ⭐⭐⭐⭐⭐

vLLM은 OpenAI API와 완전히 호환되므로, OpenAI 공식 라이브러리를 그대로 사용할 수 있습니다!

### 설치

```bash
pip install openai
```

### 기본 사용

```python
from openai import OpenAI

# vLLM 서버에 연결 (base_url만 변경)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM은 API 키 불필요
)

# 채팅 완성
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",  # vLLM에서 실행 중인 모델
    messages=[
        {"role": "user", "content": "Python이란 무엇인가요?"}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 텍스트 완성

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# 텍스트 완성 (Completion)
response = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="Python is a",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### 스트리밍

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# 스트리밍 응답
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "긴 이야기를 들려주세요"}],
    stream=True,
    max_tokens=500
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 완전한 예제

```python
# vllm_client.py
from openai import OpenAI

class VLLMClient:
    """vLLM 클라이언트 래퍼"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )
    
    def chat(self, message: str, **kwargs):
        """간단한 채팅"""
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "meta-llama/Llama-2-7b-hf"),
            messages=[{"role": "user", "content": message}],
            max_tokens=kwargs.get("max_tokens", 200),
            temperature=kwargs.get("temperature", 0.7)
        )
        return response.choices[0].message.content
    
    def stream_chat(self, message: str, **kwargs):
        """스트리밍 채팅"""
        stream = self.client.chat.completions.create(
            model=kwargs.get("model", "meta-llama/Llama-2-7b-hf"),
            messages=[{"role": "user", "content": message}],
            stream=True,
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# 사용
if __name__ == "__main__":
    client = VLLMClient()
    
    # 일반 채팅
    response = client.chat("Python의 장점은?")
    print(response)
    
    # 스트리밍
    print("\n스트리밍:")
    for chunk in client.stream_chat("인공지능에 대해 설명해주세요"):
        print(chunk, end="", flush=True)
```

---

## 2. HTTP 요청 (curl/requests)

### cURL

```bash
# 채팅 완성
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [
      {"role": "user", "content": "Python이란?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'

# 텍스트 완성
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Python is a",
    "max_tokens": 100
  }'
```

### Python requests

```python
import requests

# vLLM 서버 URL
BASE_URL = "http://localhost:8000/v1"

def chat(message: str):
    """채팅 API 호출"""
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": "meta-llama/Llama-2-7b-hf",
            "messages": [
                {"role": "user", "content": message}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
    )
    
    result = response.json()
    return result['choices'][0]['message']['content']

# 사용
response = chat("Python의 장점은?")
print(response)
```

### JavaScript/TypeScript

```javascript
// Node.js
const fetch = require('node-fetch');

async function chat(message) {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'meta-llama/Llama-2-7b-hf',
      messages: [
        { role: 'user', content: message }
      ],
      max_tokens: 200,
      temperature: 0.7
    })
  });
  
  const data = await response.json();
  return data.choices[0].message.content;
}

// 사용
chat('Python이란?').then(console.log);
```

---

## 3. LangChain

LangChain도 vLLM과 쉽게 통합됩니다.

### 설치

```bash
pip install langchain langchain-openai
```

### ChatOpenAI 사용

```python
from langchain_openai import ChatOpenAI

# vLLM을 ChatOpenAI로 사용
llm = ChatOpenAI(
    model="meta-llama/Llama-2-7b-hf",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    max_tokens=200,
    temperature=0.7
)

# 직접 호출
response = llm.invoke("Python이란 무엇인가요?")
print(response.content)
```

### Chain과 함께 사용

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# LLM 설정
llm = ChatOpenAI(
    model="meta-llama/Llama-2-7b-hf",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1"
)

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    ("user", "{question}")
])

# Chain 구성
chain = prompt | llm | StrOutputParser()

# 실행
response = chain.invoke({"question": "Python의 장점은?"})
print(response)
```

### RAG (Retrieval Augmented Generation)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# vLLM LLM
llm = ChatOpenAI(
    model="meta-llama/Llama-2-7b-hf",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1"
)

# 문서 로딩
loader = TextLoader("documents.txt")
documents = loader.load()

# 분할
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 벡터 저장소
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 질의
response = qa.invoke("문서의 주요 내용은?")
print(response)
```

---

## 4. 기타 클라이언트

### OpenAI JavaScript/TypeScript SDK

```bash
npm install openai
```

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'EMPTY'
});

async function chat(message: string) {
  const response = await client.chat.completions.create({
    model: 'meta-llama/Llama-2-7b-hf',
    messages: [{ role: 'user', content: message }],
    max_tokens: 200
  });
  
  return response.choices[0].message.content;
}

// 사용
chat('Python이란?').then(console.log);
```

### Go

```go
package main

import (
    "context"
    "fmt"
    openai "github.com/sashabaranov/go-openai"
)

func main() {
    config := openai.DefaultConfig("EMPTY")
    config.BaseURL = "http://localhost:8000/v1"
    client := openai.NewClientWithConfig(config)
    
    resp, err := client.CreateChatCompletion(
        context.Background(),
        openai.ChatCompletionRequest{
            Model: "meta-llama/Llama-2-7b-hf",
            Messages: []openai.ChatCompletionMessage{
                {
                    Role:    openai.ChatMessageRoleUser,
                    Content: "Python이란?",
                },
            },
        },
    )
    
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Println(resp.Choices[0].Message.Content)
}
```

---

## 실전 통합 예제

### 이 프로젝트 + vLLM 통합

```python
# integrated_client.py
"""
이 프로젝트로 파인튜닝한 모델을 vLLM으로 서비스
"""

from openai import OpenAI

class FinetunedLLMClient:
    """파인튜닝된 모델 클라이언트"""
    
    def __init__(
        self,
        model_path: str,
        vllm_base_url: str = "http://localhost:8000/v1"
    ):
        self.model_path = model_path
        self.client = OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY"
        )
    
    def chat(self, instruction: str, input_text: str = ""):
        """Instruction 형식으로 질의"""
        
        # Instruction 형식 프롬프트 구성
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""
        
        # vLLM 호출
        response = self.client.completions.create(
            model=self.model_path,
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            stop=["###"]  # Instruction 구분자에서 중지
        )
        
        return response.choices[0].text.strip()
    
    def batch_process(self, questions: list):
        """배치 처리"""
        results = []
        for q in questions:
            result = self.chat(q)
            results.append(result)
        return results

# 사용
if __name__ == "__main__":
    # 1. vLLM 서버 시작 (별도 터미널)
    # vllm serve outputs/my_model/final_model --port 8000
    
    # 2. 클라이언트 생성
    client = FinetunedLLMClient(
        model_path="outputs/my_model/final_model"
    )
    
    # 3. 사용
    response = client.chat(
        instruction="Python의 장점을 설명하세요"
    )
    print(response)
```

### 웹 애플리케이션 통합

```python
# app.py (FastAPI + vLLM)
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# vLLM 클라이언트
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_question(question: Question):
    """사용자 질문 처리"""
    response = vllm_client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=[
            {"role": "user", "content": question.text}
        ],
        max_tokens=200
    )
    
    return {
        "question": question.text,
        "answer": response.choices[0].message.content
    }

# 실행: uvicorn app:app --port 8080
```

---

## 성능 최적화

### 1. 배치 처리

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# 여러 요청을 동시에
questions = [
    "Python이란?",
    "머신러닝이란?",
    "딥러닝이란?"
]

responses = []
for q in questions:
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=[{"role": "user", "content": q}],
        max_tokens=100
    )
    responses.append(response.choices[0].message.content)
```

### 2. 비동기 처리

```python
import asyncio
from openai import AsyncOpenAI

async def ask_question(client, question):
    response = await client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=[{"role": "user", "content": question}],
        max_tokens=100
    )
    return response.choices[0].message.content

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    
    questions = [
        "Python이란?",
        "머신러닝이란?",
        "딥러닝이란?"
    ]
    
    # 동시 실행
    tasks = [ask_question(client, q) for q in questions]
    responses = await asyncio.gather(*tasks)
    
    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r}\n")

# 실행
asyncio.run(main())
```

---

## 트러블슈팅

### 연결 오류

```python
import requests

# vLLM 서버 확인
try:
    response = requests.get("http://localhost:8000/health")
    print("서버 정상:", response.json())
except:
    print("서버 연결 실패. vLLM 서버가 실행 중인지 확인하세요.")
```

### 모델 이름 확인

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# 사용 가능한 모델 목록
models = client.models.list()
print("사용 가능한 모델:")
for model in models.data:
    print(f"  - {model.id}")
```

---

## 요약

### 추천 순서

1. **OpenAI Python 라이브러리** ⭐⭐⭐⭐⭐
   - 가장 쉽고 강력함
   - 공식 지원
   
2. **LangChain** ⭐⭐⭐⭐
   - 복잡한 워크플로우
   - RAG, Chain 등
   
3. **HTTP 직접 호출** ⭐⭐⭐
   - 간단한 요청
   - 다른 언어

### 빠른 시작

```bash
# 1. vLLM 설치 및 시작
pip install vllm
vllm serve meta-llama/Llama-2-7b-hf --port 8000

# 2. OpenAI 라이브러리 설치
pip install openai

# 3. Python에서 사용
python
>>> from openai import OpenAI
>>> client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
>>> response = client.chat.completions.create(
...     model="meta-llama/Llama-2-7b-hf",
...     messages=[{"role": "user", "content": "Hello!"}]
... )
>>> print(response.choices[0].message.content)
```

모든 OpenAI 호환 클라이언트가 작동합니다! 🚀

