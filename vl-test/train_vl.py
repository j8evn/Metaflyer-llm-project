import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers import AutoProcessor
from PIL import Image

try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration
    MODEL_CLASS = Qwen2VLForConditionalGeneration

# 모델과 프로세서
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID)

# Dataset 정의
class ImageTextDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        text_input = item["text_input"]
        text_output = item["text_output"]

        # 이미지 처리
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Dummy image for robustness
            image = Image.new("RGB", (224, 224), color="black")

        # 텍스트 처리 (Qwen2-VL style)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_input},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Prompt 길이 계산을 위한 처리
        prompt_inputs = processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        prompt_length = prompt_inputs.input_ids.shape[1]

        full_text = text + text_output + "<|endoftext|>" 
        
        inputs = processor(
            text=[full_text],
            images=[image],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw if "image_grid_thw" in inputs else None
        
        # Label Masking
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Prompt 부분 마스킹
        labels[input_ids == processor.tokenizer.pad_token_id] = -100 # Padding 부분 마스킹
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }

# Dataset 로드
train_dataset = ImageTextDataset("data/dataset.json")

# 커스텀 collate_fn
def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    
    pixel_values = [b["pixel_values"] for b in batch if b["pixel_values"] is not None]
    if pixel_values:
        pixel_values = torch.cat(pixel_values, dim=0)
    else:
        pixel_values = None

    image_grid_thw = [b["image_grid_thw"] for b in batch if b["image_grid_thw"] is not None]
    if image_grid_thw:
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
    else:
        image_grid_thw = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True,
    fp16=False,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

# 학습 시작
trainer.train()

