import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import os

# 설정
BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
ADAPTER_PATH = "./output/checkpoint-21" # 학습된 LoRA 어댑터 경로 (가장 최근 체크포인트 사용 권장, 예: ./output/checkpoint-500)
OUTPUT_DIR = "./merged_model"

print(f"Loading base model: {BASE_MODEL_ID}")
# Base Model 로드
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
# LoRA Adapter 로드 및 병합
# 주의: output 폴더 안에 adapter_model.bin 또는 safetensors가 있어야 합니다.
# 만약 checkpoint 폴더 안에 있다면 ADAPTER_PATH를 해당 폴더로 수정하세요.
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
except Exception as e:
    print(f"Error loading adapter: {e}")
    print("Tip: ADAPTER_PATH가 정확한지 확인하세요. (예: ./output/checkpoint-100)")
    exit(1)

print("Merging model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("Done!")
