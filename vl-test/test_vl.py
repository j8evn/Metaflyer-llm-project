import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from PIL import Image

model_name = "Qwen/Qwen3-VL-4B-Instruct"
lora_path = "./qwen3-vl-lora"

# Processor
processor = AutoProcessor.from_pretrained(lora_path)

# Base model + LoRA merge
base = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, lora_path)
model.eval()

# Test image
img = Image.open("test.jpg")

prompt = "이 이미지에 대해 자세히 설명해줘."

inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200
    )

print(processor.decode(output[0], skip_special_tokens=True))

