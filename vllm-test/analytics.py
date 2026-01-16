import os
import json
import glob
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

VALIDATION_BASE_DIR = "/dataset/cep/37.한국적_영상_이해_데이터/3.개방데이터/1.데이터/Validation_Sample_1000"
IMG_DIR = os.path.join(VALIDATION_BASE_DIR, "01.원천데이터")
OUTPUT_FILE = "extracted_metadata.json"

# 우선 원본 모델로 테스트 (학습 모델이 ./merged_model에 생기면 변경)
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
ADAPTER_PATH = "./qwen3_vl_lora_results/final_adapter" # Added ADAPTER_PATH

print("=" * 60)
print("VidBrick-VL v1.0: 이미지 키워드 추출 (Transformers Direct)")
print("=" * 60)

# 1. 모델 및 프로세서 로드
print("\n[1/3] 모델 및 프로세서 로드 중 (Loading with 4-bit optimization)...")
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# GPU 1, 2, 3번에 분산 로딩 (각 70GB 제한으로 여유 공간 확보)
max_memory = {0: "70GiB", 1: "70GiB", 2: "70GiB"}

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

if os.path.exists(ADAPTER_PATH): # Added LoRA adapter loading logic
    print(f"어댑터 로드 중: {ADAPTER_PATH}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("어댑터 적용 완료")
else:
    print("어댑터를 찾을 수 없어 기본 모델로 진행합니다.")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("로드 완료")

# 2. 이미지 파일 수집 (1,000개 샘플링)
print("\n[2/3] 테스트 데이터 수집 중 (Sampling 1,000 files)...")
all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
image_files = all_images[:1000] # 임시 3개 테스트 종료 -> 공식 1,000개 시험으로 복구
print(f"{len(image_files)}개 파일 준비 완료")

# 3. 추론 (Inference)
print("\n[3/3] 키워드 추출 시작...")
results = {}

for img_path in tqdm(image_files, desc="Processing"):
    try:
        filename = os.path.basename(img_path).replace(".png", "")
        
        # 메시지 구성 (Qwen3-VL 정석 포맷)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": "이 이미지에서 핵심 키워드를 추출해줘."},
                ],
            }
        ]

        # 전처리
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # 생성
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 따옴표 및 불필요한 공백 철저히 제거
            cleaned_keywords = [k.strip().replace("'", "").replace('"', "") for k in output_text.split(",")]
            results[filename] = {"keywords": cleaned_keywords}


    except Exception as e:
        print(f"\n⚠ 에러 발생 ({filename}): {str(e)}")
        results[filename] = {"keywords": []}

# 4. 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"완료! 결과가 {OUTPUT_FILE}에 저장되었습니다.")
print("=" * 60)
