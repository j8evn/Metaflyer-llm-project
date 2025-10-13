"""
모델 양자화 스크립트
학습된 모델을 양자화하여 크기와 메모리 사용량을 줄입니다.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append('../src')


def quantize_model(
    model_path: str,
    output_path: str,
    bits: int = 8,
    save_format: str = "safetensors"
):
    """
    모델 양자화
    
    Args:
        model_path: 원본 모델 경로
        output_path: 양자화된 모델 저장 경로
        bits: 양자화 비트 수 (4 또는 8)
        save_format: 저장 형식 (safetensors 또는 bin)
    """
    print("=" * 60)
    print(f"{bits}-bit 모델 양자화 시작")
    print("=" * 60)
    
    print(f"\n[1/3] 모델 로딩: {model_path}")
    
    if bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError("bits는 4 또는 8이어야 합니다")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"\n[2/3] 양자화된 모델 저장: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    model.save_pretrained(
        output_path,
        safe_serialization=(save_format == "safetensors")
    )
    tokenizer.save_pretrained(output_path)
    
    print(f"\n[3/3] 완료!")
    
    # 모델 크기 비교
    original_size = get_directory_size(model_path)
    quantized_size = get_directory_size(output_path)
    
    print("\n" + "=" * 60)
    print("양자화 완료!")
    print("=" * 60)
    print(f"\n원본 모델 크기: {original_size / (1024**3):.2f} GB")
    print(f"양자화된 모델 크기: {quantized_size / (1024**3):.2f} GB")
    print(f"압축률: {(1 - quantized_size/original_size) * 100:.1f}%")
    print(f"\n저장 경로: {output_path}")


def get_directory_size(path: str) -> int:
    """디렉토리 크기 계산 (바이트)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def main():
    parser = argparse.ArgumentParser(description="모델 양자화 스크립트")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="원본 모델 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="양자화된 모델 저장 경로"
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="양자화 비트 수 (4 또는 8)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["safetensors", "bin"],
        default="safetensors",
        help="저장 형식"
    )
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("경고: CUDA를 사용할 수 없습니다. 양자화가 CPU에서 실행되어 느릴 수 있습니다.")
    
    quantize_model(
        model_path=args.model_path,
        output_path=args.output,
        bits=args.bits,
        save_format=args.format
    )


if __name__ == "__main__":
    main()

