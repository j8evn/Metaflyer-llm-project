import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import os

# 설정
BASE_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
ADAPTER_PATH = None 
OUTPUT_DIR = "./merged_model"

# 자동으로 어댑터 경로 찾기 (final_adapter 우선, 없으면 최신 checkpoint)
if ADAPTER_PATH is None:
    import glob
    final_path = "./output_full/final_adapter"
    checkpoints = glob.glob("./output_full/checkpoint-*")
    
    if os.path.exists(final_path):
        ADAPTER_PATH = final_path
        print(f"Detected final adapter: {ADAPTER_PATH}")
    elif checkpoints:
        ADAPTER_PATH = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"Auto-detected latest checkpoint: {ADAPTER_PATH}")
    else:
        print("Error: No adapter or checkpoints found in ./output_full")
        exit(1)

print(f"Loading base model: {BASE_MODEL_ID}")
# Base Model 로드 (메모리 절약을 위해 bf16 및 low_cpu_mem_usage 사용)
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # 병합은 보통 CPU에서 메모리를 많이 쓰므로 명시적 지정 혹은 auto
    low_cpu_mem_usage=True
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
