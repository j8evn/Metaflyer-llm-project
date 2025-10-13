"""
API 서버 테스트 스크립트
"""

import requests
import json
import time


def test_health(base_url: str = "http://localhost:8000"):
    """헬스체크 테스트"""
    print("=" * 60)
    print("헬스체크 테스트")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/health")
    print(f"상태 코드: {response.status_code}")
    print(f"응답: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_generate(base_url: str = "http://localhost:8000"):
    """텍스트 생성 테스트"""
    print("=" * 60)
    print("텍스트 생성 테스트")
    print("=" * 60)
    
    data = {
        "prompt": "Python이란 무엇인가요?",
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    print(f"요청 데이터: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print()
    
    start_time = time.time()
    response = requests.post(f"{base_url}/generate", json=data)
    elapsed_time = time.time() - start_time
    
    print(f"상태 코드: {response.status_code}")
    print(f"API 응답 시간: {elapsed_time:.2f}초")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n생성된 텍스트:")
        print("-" * 60)
        for i, text in enumerate(result['generated_text'], 1):
            print(f"[{i}] {text}")
            print("-" * 60)
        print(f"\n모델: {result['model_name']}")
        print(f"생성 시간: {result['generation_time']:.2f}초")
    else:
        print(f"오류: {response.text}")
    
    print()


def test_chat(base_url: str = "http://localhost:8000"):
    """대화형 생성 테스트"""
    print("=" * 60)
    print("대화형 생성 테스트")
    print("=" * 60)
    
    data = {
        "instruction": "Python에서 리스트와 튜플의 차이점을 설명하세요",
        "input": "",
        "max_new_tokens": 150,
        "temperature": 0.7
    }
    
    print(f"요청 데이터: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print()
    
    start_time = time.time()
    response = requests.post(f"{base_url}/chat", json=data)
    elapsed_time = time.time() - start_time
    
    print(f"상태 코드: {response.status_code}")
    print(f"API 응답 시간: {elapsed_time:.2f}초")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n응답:")
        print("-" * 60)
        print(result['response'])
        print("-" * 60)
        print(f"\n모델: {result['model_name']}")
        print(f"생성 시간: {result['generation_time']:.2f}초")
    else:
        print(f"오류: {response.text}")
    
    print()


def test_model_info(base_url: str = "http://localhost:8000"):
    """모델 정보 테스트"""
    print("=" * 60)
    print("모델 정보 테스트")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/model_info")
    print(f"상태 코드: {response.status_code}")
    print(f"응답: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def interactive_test(base_url: str = "http://localhost:8000"):
    """대화형 테스트"""
    print("=" * 60)
    print("대화형 테스트 (종료: 'quit', 'exit', 'q')")
    print("=" * 60)
    print()
    
    while True:
        try:
            instruction = input("질문을 입력하세요: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                print("테스트를 종료합니다.")
                break
            
            if not instruction:
                continue
            
            data = {
                "instruction": instruction,
                "input": "",
                "max_new_tokens": 200,
                "temperature": 0.7
            }
            
            print("\n생성 중...")
            response = requests.post(f"{base_url}/chat", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n응답:\n{result['response']}\n")
                print(f"(생성 시간: {result['generation_time']:.2f}초)\n")
            else:
                print(f"오류: {response.text}\n")
        
        except KeyboardInterrupt:
            print("\n\n테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="API 서버 테스트")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000",
        help="API 서버 URL"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "health", "generate", "chat", "info", "interactive"],
        default="all",
        help="테스트 모드"
    )
    
    args = parser.parse_args()
    
    print(f"\nAPI 서버: {args.base_url}\n")
    
    if args.mode == "all":
        test_health(args.base_url)
        test_model_info(args.base_url)
        test_generate(args.base_url)
        test_chat(args.base_url)
    elif args.mode == "health":
        test_health(args.base_url)
    elif args.mode == "generate":
        test_generate(args.base_url)
    elif args.mode == "chat":
        test_chat(args.base_url)
    elif args.mode == "info":
        test_model_info(args.base_url)
    elif args.mode == "interactive":
        interactive_test(args.base_url)


if __name__ == "__main__":
    main()

