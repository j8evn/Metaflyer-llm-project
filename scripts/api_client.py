"""
API 클라이언트 예제
Python에서 API를 호출하는 방법
"""

import requests
from typing import List, Optional


class LLMClient:
    """LLM API 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Args:
            base_url: API 서버 URL
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """헬스체크"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> dict:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_p: Top-p 샘플링
            top_k: Top-k 샘플링
            repetition_penalty: 반복 패널티
            do_sample: 샘플링 사용 여부
            num_return_sequences: 생성할 응답 수
        
        Returns:
            생성 결과 딕셔너리
        """
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences
        }
        
        response = requests.post(f"{self.base_url}/generate", json=data)
        response.raise_for_status()
        return response.json()
    
    def chat(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> dict:
        """
        대화형 생성 (Instruction 형식)
        
        Args:
            instruction: 질문 또는 지시사항
            input_text: 추가 입력 (선택사항)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_p: Top-p 샘플링
            top_k: Top-k 샘플링
            repetition_penalty: 반복 패널티
        
        Returns:
            대화 결과 딕셔너리
        """
        data = {
            "instruction": instruction,
            "input": input_text,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }
        
        response = requests.post(f"{self.base_url}/chat", json=data)
        response.raise_for_status()
        return response.json()
    
    def model_info(self) -> dict:
        """모델 정보 조회"""
        response = requests.get(f"{self.base_url}/model_info")
        response.raise_for_status()
        return response.json()


# ============== 사용 예제 ==============

def example_basic():
    """기본 사용 예제"""
    print("=" * 60)
    print("기본 사용 예제")
    print("=" * 60)
    
    # 클라이언트 생성
    client = LLMClient("http://localhost:8000")
    
    # 헬스체크
    health = client.health_check()
    print(f"서버 상태: {health['status']}")
    print(f"모델 로딩: {health['model_loaded']}")
    print()
    
    # 텍스트 생성
    result = client.generate(
        prompt="Python이란 무엇인가요?",
        max_new_tokens=100,
        temperature=0.7
    )
    
    print(f"생성된 텍스트:")
    print(result['generated_text'][0])
    print()


def example_chat():
    """대화 예제"""
    print("=" * 60)
    print("대화 예제")
    print("=" * 60)
    
    client = LLMClient("http://localhost:8000")
    
    # 대화형 생성
    result = client.chat(
        instruction="Python에서 리스트를 정렬하는 방법을 알려주세요",
        max_new_tokens=150
    )
    
    print(f"질문: {result['instruction']}")
    print(f"응답: {result['response']}")
    print(f"생성 시간: {result['generation_time']:.2f}초")
    print()


def example_batch():
    """배치 처리 예제"""
    print("=" * 60)
    print("배치 처리 예제")
    print("=" * 60)
    
    client = LLMClient("http://localhost:8000")
    
    questions = [
        "Python이란?",
        "머신러닝이란?",
        "딥러닝이란?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}] 질문: {question}")
        result = client.chat(instruction=question, max_new_tokens=100)
        print(f"답변: {result['response'][:100]}...")


def example_error_handling():
    """에러 처리 예제"""
    print("=" * 60)
    print("에러 처리 예제")
    print("=" * 60)
    
    client = LLMClient("http://localhost:8000")
    
    try:
        result = client.chat(
            instruction="테스트 질문",
            max_new_tokens=100
        )
        print(f"성공: {result['response']}")
        
    except requests.exceptions.ConnectionError:
        print("오류: API 서버에 연결할 수 없습니다")
    except requests.exceptions.HTTPError as e:
        print(f"오류: HTTP {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"오류: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "basic":
            example_basic()
        elif mode == "chat":
            example_chat()
        elif mode == "batch":
            example_batch()
        elif mode == "error":
            example_error_handling()
        else:
            print(f"사용법: python api_client.py [basic|chat|batch|error]")
    else:
        print("모든 예제 실행:")
        example_basic()
        example_chat()
        example_batch()

