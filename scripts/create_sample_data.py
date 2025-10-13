"""
샘플 데이터 생성 스크립트
다양한 형식의 샘플 데이터를 생성합니다.
"""

import json
import os
import argparse
from pathlib import Path


def create_instruction_dataset(output_path: str, num_samples: int = 100):
    """Instruction-Input-Output 형식의 데이터셋 생성"""
    
    sample_data = [
        {
            "instruction": "다음 문장을 한국어로 번역하세요.",
            "input": "Hello, how are you?",
            "output": "안녕하세요, 어떻게 지내세요?"
        },
        {
            "instruction": "다음 질문에 답하세요.",
            "input": "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
            "output": "리스트는 변경 가능한(mutable) 자료구조이고, 튜플은 변경 불가능한(immutable) 자료구조입니다. 리스트는 대괄호 []를 사용하고, 튜플은 소괄호 ()를 사용합니다."
        },
        {
            "instruction": "다음 코드의 결과를 예측하세요.",
            "input": "x = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)",
            "output": "[1, 2, 3, 4]가 출력됩니다. 리스트는 참조로 전달되므로 y를 수정하면 x도 함께 변경됩니다."
        },
        {
            "instruction": "간단한 자기소개를 작성하세요.",
            "input": "",
            "output": "안녕하세요! 저는 다양한 주제에 대해 도움을 드릴 수 있는 AI 어시스턴트입니다. 질문이나 요청사항이 있으시면 언제든지 말씀해 주세요."
        },
        {
            "instruction": "다음 수학 문제를 풀어주세요.",
            "input": "25 × 4 = ?",
            "output": "25 × 4 = 100입니다."
        },
        {
            "instruction": "다음 단어의 반대말을 말하세요.",
            "input": "크다",
            "output": "작다"
        },
        {
            "instruction": "다음 문장에서 문법 오류를 찾아 수정하세요.",
            "input": "나는 어제 학교에 갔었어요.",
            "output": "나는 어제 학교에 갔어요. ('갔었어요'는 이중 과거 표현으로 불필요합니다)"
        },
        {
            "instruction": "다음 주제에 대해 짧게 설명하세요.",
            "input": "인공지능",
            "output": "인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다. 머신러닝, 딥러닝 등의 기술을 통해 데이터로부터 패턴을 학습하고 의사결정을 수행할 수 있습니다."
        }
    ]
    
    # 샘플 반복하여 요청된 수만큼 생성
    full_data = []
    for i in range(num_samples):
        sample = sample_data[i % len(sample_data)].copy()
        full_data.append(sample)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Instruction 형식 데이터셋 생성 완료: {output_path} ({num_samples} 샘플)")


def create_chat_dataset(output_path: str, num_samples: int = 50):
    """Chat/대화 형식의 데이터셋 생성"""
    
    sample_conversations = [
        {
            "messages": [
                {"role": "user", "content": "안녕하세요!"},
                {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
                {"role": "user", "content": "Python 프로그래밍에 대해 알려주세요."},
                {"role": "assistant", "content": "Python은 배우기 쉽고 강력한 프로그래밍 언어입니다. 웹 개발, 데이터 분석, 인공지능 등 다양한 분야에서 활용됩니다."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "오늘 날씨가 좋네요."},
                {"role": "assistant", "content": "네, 맞습니다! 좋은 날씨를 즐기시기 바랍니다."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "머신러닝과 딥러닝의 차이는?"},
                {"role": "assistant", "content": "머신러닝은 데이터로부터 패턴을 학습하는 기술 전반을 의미하며, 딥러닝은 신경망을 사용하는 머신러닝의 한 분야입니다. 딥러닝은 더 복잡한 패턴을 학습할 수 있습니다."}
            ]
        }
    ]
    
    full_data = []
    for i in range(num_samples):
        sample = sample_conversations[i % len(sample_conversations)].copy()
        full_data.append(sample)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Chat 형식 데이터셋 생성 완료: {output_path} ({num_samples} 샘플)")


def create_text_dataset(output_path: str, num_samples: int = 100):
    """단순 텍스트 형식의 데이터셋 생성"""
    
    sample_texts = [
        {
            "text": "Python은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어입니다. 간결하고 읽기 쉬운 문법으로 초보자부터 전문가까지 널리 사용됩니다."
        },
        {
            "text": "인공지능 기술은 의료, 금융, 제조 등 다양한 산업 분야에서 혁신을 이끌고 있습니다. 특히 딥러닝 기술의 발전으로 이미지 인식, 자연어 처리 등에서 눈부신 성과를 거두고 있습니다."
        },
        {
            "text": "오픈 소스 소프트웨어는 소스 코드가 공개되어 누구나 자유롭게 사용, 수정, 배포할 수 있는 소프트웨어입니다. Linux, Python, TensorFlow 등이 대표적인 예입니다."
        }
    ]
    
    full_data = []
    for i in range(num_samples):
        sample = sample_texts[i % len(sample_texts)].copy()
        full_data.append(sample)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Text 형식 데이터셋 생성 완료: {output_path} ({num_samples} 샘플)")


def main():
    parser = argparse.ArgumentParser(description="샘플 데이터셋 생성")
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["instruction", "chat", "text", "all"],
        default="all",
        help="생성할 데이터 형식"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=100,
        help="학습 샘플 수"
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=20,
        help="평가 샘플 수"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("샘플 데이터셋 생성 시작")
    print("=" * 50)
    
    if args.format in ["instruction", "all"]:
        create_instruction_dataset(
            f"{args.output_dir}/train_instruction.json",
            args.num_train
        )
        create_instruction_dataset(
            f"{args.output_dir}/eval_instruction.json",
            args.num_eval
        )
    
    if args.format in ["chat", "all"]:
        create_chat_dataset(
            f"{args.output_dir}/train_chat.json",
            args.num_train // 2
        )
    
    if args.format in ["text", "all"]:
        create_text_dataset(
            f"{args.output_dir}/train_text.json",
            args.num_train
        )
    
    # 기본 train.json과 eval.json도 생성 (instruction 형식)
    if args.format == "all":
        create_instruction_dataset(
            f"{args.output_dir}/train.json",
            args.num_train
        )
        create_instruction_dataset(
            f"{args.output_dir}/eval.json",
            args.num_eval
        )
    
    print("=" * 50)
    print("샘플 데이터셋 생성 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()

