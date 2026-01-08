import json
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoProcessor
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType

# 1. 모델 설정 (H100용 고성능 설정)
MODEL_ID = "Qwen/Qwen3-VL-30B-Instruct"

try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration
    MODEL_CLASS = Qwen2VLForConditionalGeneration

# 모델 로드 (bfloat16 사용)
print(f"Loading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MODEL_CLASS.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto", # H100에서는 auto로 설정해도 충분함 (Trainer가 알아서 처리하지만 명시적으로도 무방)
)

# 2. LoRA 설정 (성능 중심)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,              # Rank 대폭 상향 (학습 용량 증대)
    lora_alpha=128,    # Alpha도 상향
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 모든 선형 레이어 학습
)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Dataset 정의
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
            # H100이므로 이미지 리사이즈 제한 제거 (원본 해상도 활용)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        # 텍스트 처리
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
        
        # 입력 처리
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        
        # 정답 라벨 생성
        full_text = text + text_output + "<|endoftext|>"
        inputs_full = processor(
            text=[full_text],
            images=[image],
            padding="max_length",
            max_length=1024, # Max Length 증가
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = inputs_full.input_ids.squeeze(0)
        attention_mask = inputs_full.attention_mask.squeeze(0)
        labels = input_ids.clone()
        
        # Prompt 부분 마스킹 (질문은 학습하지 않음)
        prompt_length = inputs.input_ids.shape[1]
        labels[:prompt_length] = -100
        labels[input_ids == processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs_full.pixel_values,
            "image_grid_thw": inputs_full.image_grid_thw if "image_grid_thw" in inputs_full else None
        }

train_dataset = ImageTextDataset("data/dataset.json")

# 4. Collate Function
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
    pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None

    image_grid_thw = [b["image_grid_thw"] for b in batch if b["image_grid_thw"] is not None]
    image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }

# 5. Training Arguments (H100 최적화)
training_args = TrainingArguments(
    output_dir="./output_full",
    overwrite_output_dir=True, # 기존 학습 결과 덮어쓰기
    per_device_train_batch_size=4,  # 배치 사이즈 증가
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,    # 메모리 여유가 있어도 켜두는 게 안전 (속도 차이 크지 않음)
    num_train_epochs=5,             # Epoch 증가
    learning_rate=2e-5,             # LoRA Rank가 높으므로 LR 조정
    bf16=True,                      # H100 필수 설정
    optim="adamw_torch",            # 표준 AdamW 사용 (Adafactor보다 성능 좋음)
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False,    # MPS 경고 방지용이었으나 Linux에서는 True여도 무방 (여기선 안전하게 False 유지)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

print("Starting High-Performance Training on H100...")
trainer.train()
