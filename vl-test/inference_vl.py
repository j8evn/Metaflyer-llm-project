import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import requests

# 설정
MODEL_PATH = "./merged_model" # 병합된 모델 경로
IMAGE_PATH = "data/test.jpg" # 테스트할 이미지 경로 (변경 필요)
PROMPT = "이 인물은 누구입니까?"

print(f"Loading model from: {MODEL_PATH}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 이미지 로드 (로컬 파일 또는 URL)
try:
    if IMAGE_PATH.startswith("http"):
        image = Image.open(requests.get(IMAGE_PATH, stream=True).raw)
    else:
        image = Image.open(IMAGE_PATH)
except Exception as e:
    print(f"Error loading image: {e}")
    print("Using dummy image for testing...")
    image = Image.new("RGB", (224, 224), color="white")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
]

# Device 설정
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
model.to(device)

# Inference 준비
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

# 생성
print("Generating...")
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("-" * 30)
print(f"Output: {output_text[0]}")
print("-" * 30)
