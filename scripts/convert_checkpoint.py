"""
LoRA 체크포인트를 전체 모델로 병합하는 스크립트
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append('../src')


def merge_lora_weights(
    base_model_path: str,
    lora_model_path: str,
    output_path: str
):
    """
    LoRA 가중치를 베이스 모델과 병합
    
    Args:
        base_model_path: 베이스 모델 경로
        lora_model_path: LoRA 어댑터 경로
        output_path: 병합된 모델 저장 경로
    """
    print("=" * 60)
    print("LoRA 가중치 병합 시작")
    print("=" * 60)
    
    print(f"\n[1/4] 베이스 모델 로딩: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"\n[2/4] LoRA 어댑터 로딩: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    print(f"\n[3/4] 가중치 병합 중...")
    model = model.merge_and_unload()
    
    print(f"\n[4/4] 병합된 모델 저장: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    # 토크나이저도 함께 저장
    print("토크나이저 저장 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("\n" + "=" * 60)
    print("병합 완료!")
    print("=" * 60)
    print(f"\n병합된 모델 경로: {output_path}")
    print("이제 이 모델을 일반 Hugging Face 모델처럼 사용할 수 있습니다.")


def main():
    parser = argparse.ArgumentParser(description="LoRA 체크포인트 병합 스크립트")
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="베이스 모델 경로 또는 Hugging Face ID"
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        required=True,
        help="LoRA 어댑터 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="병합된 모델 저장 경로"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_path=args.base_model,
        lora_model_path=args.lora_model,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

