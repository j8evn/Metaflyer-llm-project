"""
vLLM 클라이언트 사용 예제
OpenAI 라이브러리를 사용하여 vLLM에 접근
"""

from openai import OpenAI
import asyncio
from openai import AsyncOpenAI


# ============== 기본 사용 ==============

def basic_example():
    """기본 채팅 예제"""
    print("=" * 60)
    print("1. 기본 채팅 예제")
    print("=" * 60)
    
    # vLLM 클라이언트 생성
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
    
    print(f"\n질문: Python이란 무엇인가요?")
    print(f"답변: {response.choices[0].message.content}")
    print()


# ============== 스트리밍 ==============

def streaming_example():
    """스트리밍 응답 예제"""
    print("=" * 60)
    print("2. 스트리밍 응답 예제")
    print("=" * 60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    
    print("\n질문: 인공지능에 대해 설명해주세요")
    print("답변: ", end="", flush=True)
    
    # 스트리밍
    stream = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=[
            {"role": "user", "content": "인공지능에 대해 설명해주세요"}
        ],
        stream=True,
        max_tokens=200
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


# ============== 대화 히스토리 ==============

def conversation_example():
    """대화 히스토리 예제"""
    print("=" * 60)
    print("3. 대화 히스토리 예제")
    print("=" * 60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    
    # 대화 히스토리
    messages = [
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
        {"role": "user", "content": "안녕하세요!"},
    ]
    
    # 첫 번째 응답
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=messages,
        max_tokens=100
    )
    
    assistant_message = response.choices[0].message.content
    print(f"User: 안녕하세요!")
    print(f"AI: {assistant_message}\n")
    
    # 히스토리에 추가
    messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": "Python에 대해 알려주세요"})
    
    # 두 번째 응답 (히스토리 포함)
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=messages,
        max_tokens=150
    )
    
    print(f"User: Python에 대해 알려주세요")
    print(f"AI: {response.choices[0].message.content}\n")


# ============== 비동기 예제 ==============

async def async_example():
    """비동기 요청 예제"""
    print("=" * 60)
    print("4. 비동기 요청 예제")
    print("=" * 60)
    
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    
    # 여러 질문을 동시에 처리
    questions = [
        "Python이란?",
        "머신러닝이란?",
        "딥러닝이란?"
    ]
    
    async def ask(question):
        response = await client.chat.completions.create(
            model="meta-llama/Llama-2-7b-hf",
            messages=[{"role": "user", "content": question}],
            max_tokens=100
        )
        return question, response.choices[0].message.content
    
    # 동시 실행
    print("\n여러 질문을 동시에 처리...\n")
    tasks = [ask(q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    for question, answer in results:
        print(f"Q: {question}")
        print(f"A: {answer}\n")


# ============== 클래스 래퍼 ==============

class VLLMClient:
    """vLLM 클라이언트 래퍼 클래스"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-2-7b-hf"
    ):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
    
    def chat(self, message: str, **kwargs):
        """간단한 채팅"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            max_tokens=kwargs.get("max_tokens", 200),
            temperature=kwargs.get("temperature", 0.7)
        )
        return response.choices[0].message.content
    
    def chat_with_history(self, messages: list, **kwargs):
        """히스토리를 포함한 채팅"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 200),
            temperature=kwargs.get("temperature", 0.7)
        )
        return response.choices[0].message.content
    
    def stream_chat(self, message: str, **kwargs):
        """스트리밍 채팅"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            stream=True,
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def class_wrapper_example():
    """클래스 래퍼 예제"""
    print("=" * 60)
    print("5. 클래스 래퍼 예제")
    print("=" * 60)
    
    # 클라이언트 생성
    client = VLLMClient()
    
    # 간단한 채팅
    print("\n간단한 채팅:")
    response = client.chat("Python의 장점은?")
    print(f"답변: {response}\n")
    
    # 스트리밍
    print("스트리밍:")
    print("답변: ", end="", flush=True)
    for chunk in client.stream_chat("딥러닝에 대해 설명해주세요"):
        print(chunk, end="", flush=True)
    print("\n")


# ============== 파인튜닝된 모델 사용 ==============

def finetuned_model_example():
    """파인튜닝된 모델 사용 예제"""
    print("=" * 60)
    print("6. 파인튜닝된 모델 사용 예제")
    print("=" * 60)
    
    # 파인튜닝된 모델의 경우 Instruction 형식 사용
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    
    # Instruction 형식 프롬프트
    instruction = "Python에서 리스트를 정렬하는 방법을 알려주세요"
    prompt = f"""### Instruction:
{instruction}

### Response:
"""
    
    response = client.completions.create(
        model="outputs/my_model/final_model",  # 파인튜닝한 모델
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        stop=["###"]  # Instruction 구분자에서 중지
    )
    
    print(f"\nInstruction: {instruction}")
    print(f"Response: {response.choices[0].text.strip()}\n")


# ============== 메인 ==============

def main():
    """모든 예제 실행"""
    print("\nvLLM 클라이언트 예제")
    print("=" * 60)
    print("시작하기 전에 vLLM 서버가 실행 중인지 확인하세요:")
    print("  vllm serve meta-llama/Llama-2-7b-hf --port 8000")
    print("=" * 60)
    print()
    
    try:
        # 1. 기본 예제
        basic_example()
        
        # 2. 스트리밍
        streaming_example()
        
        # 3. 대화 히스토리
        conversation_example()
        
        # 4. 비동기 (선택사항)
        print("비동기 예제 실행 중...\n")
        asyncio.run(async_example())
        
        # 5. 클래스 래퍼
        class_wrapper_example()
        
        # 6. 파인튜닝 모델 (선택사항)
        # finetuned_model_example()
        
        print("=" * 60)
        print("모든 예제 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("\nvLLM 서버가 실행 중인지 확인하세요:")
        print("  vllm serve meta-llama/Llama-2-7b-hf --port 8000")


if __name__ == "__main__":
    main()

