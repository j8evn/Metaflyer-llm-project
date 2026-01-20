"""
Qwen3-VL 이미지 캡셔닝 LoRA 어댑터 병합 스크립트

사용법:
    # 기본 (output_caption에서 자동 탐색)
    python merge_caption_lora.py

    # 테스트 모드 결과 병합
    python merge_caption_lora.py --test

    # 직접 경로 지정
    python merge_caption_lora.py --adapter_path ./output_caption_test/final_adapter
"""

import argparse
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import os
import glob

# 설정
HF_CACHE_DIR = "/dataset/cep/cache/huggingface/hub"
BASE_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def find_adapter_path(base_dir: str) -> str:
    """어댑터 경로 자동 탐색"""
    # final_adapter 확인
    final_adapter = os.path.join(base_dir, "final_adapter")
    if os.path.exists(final_adapter):
        return final_adapter

    # 체크포인트에서 찾기
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if checkpoints:
        # 숫자 기준으로 정렬 (checkpoint-100, checkpoint-200 ...)
        return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    return None


def merge(args):
    # 어댑터 경로 결정
    if args.adapter_path:
        adapter_path = args.adapter_path
    else:
        # 테스트 모드 또는 기본 경로에서 탐색
        base_dir = "./output_caption_test" if args.test else "./output_caption"
        adapter_path = find_adapter_path(base_dir)

        if not adapter_path:
            print(f"Error: No adapter found in {base_dir}")
            print("Please run train_caption_lora.py first.")
            exit(1)

    # 출력 경로 결정
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./merged_caption_model_test" if args.test else "./merged_caption_model"

    print("=" * 60)
    print("Qwen3-VL LoRA 어댑터 병합")
    print("=" * 60)
    print(f"Adapter: {adapter_path}")
    print(f"Output:  {output_dir}")

    # Base Model 로드
    print(f"\n[1/3] Base 모델 로드 중: {BASE_MODEL_ID}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=HF_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE_DIR)

    # LoRA Adapter 로드
    print(f"\n[2/3] LoRA 어댑터 로드 중: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print(f"Tip: Check if {adapter_path} exists and contains adapter_model.safetensors")
        exit(1)

    # 병합
    print("\n[3/3] 모델 병합 중...")
    model = model.merge_and_unload()

    # 저장
    print(f"\n모델 저장 중: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print("병합 완료!")
    print(f"병합된 모델 위치: {output_dir}")
    print("\nvLLM 서빙 명령어:")
    print(f"  vllm serve {output_dir} --host 0.0.0.0 --port 8100 --trust-remote-code")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL LoRA 어댑터 병합")

    parser.add_argument("--test", action="store_true",
                        help="테스트 모드 결과 병합 (output_caption_test)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA 어댑터 경로 (직접 지정)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="병합된 모델 저장 경로")

    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()

