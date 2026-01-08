import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import os

# 설정
BASE_MODEL_ID = "Qwen/Qwen3-VL-30B-Instruct"
ADAPTER_PATH = None # None으로 두면 output_full에서 가장 최신 체크포인트를 자동으로 찾습니다.
OUTPUT_DIR = "./merged_model"

# 자동으로 최신 체크포인트 찾기
if ADAPTER_PATH is None:
    import glob
    checkpoints = glob.glob("./output_full/checkpoint-*")
    if not checkpoints:
        print("Error: No checkpoints found in ./output_full")
        exit(1)
    # 숫자 기준으로 정렬 (checkpoint-100, checkpoint-200 ...)
    ADAPTER_PATH = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(f"Auto-detected latest checkpoint: {ADAPTER_PATH}")

print(f"Loading base model: {BASE_MODEL_ID}")
# Base Model 로드
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
# LoRA Adapter 로드 및 병합
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
except Exception as e:
    print(f"Error loading adapter: {e}")
    print(f"Tip: Check if {ADAPTER_PATH} exists and contains adapter_model.safetensors")
    exit(1)

print("Merging model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("Done!")
