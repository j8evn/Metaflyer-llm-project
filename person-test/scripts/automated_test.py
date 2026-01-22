import os
import requests
import json
import base64
import time
from tqdm import tqdm

# vLLM Server Configuration
API_URL = "http://localhost:18001/v1/chat/completions"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(BASE_DIR, "data/test")
PEOPLE_LIST_JSON = os.path.join(BASE_DIR, "config/people.json")
MODEL_NAME = "/dataset/cep/llm-training/person-test/merged_model"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_target_people():
    if os.path.exists(PEOPLE_LIST_JSON):
        with open(PEOPLE_LIST_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def test_inference(image_path):
    base64_image = encode_image(image_path)
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": "이 인물은 누구입니까?"}
                ]
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0 # 일관성을 위해 0으로 설정
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    target_people = load_target_people()
    if not target_people:
        print("No target people found in config/people.json")
        return

    results = []
    total_images = 0
    correct_count = 0

    print(f"Starting automated test on {TEST_DATA_DIR}...")
    
    # 각 인물 폴더별로 테스트 진행
    for name, query in tqdm(target_people.items(), desc="People"):
        person_dir = os.path.join(TEST_DATA_DIR, query.replace(" ", "_"))
        if not os.path.exists(person_dir):
            continue
            
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            prediction = test_inference(img_path)
            
            # 정답 판정 (모델 출력에 실제 이름이 포함되어 있는지 확인)
            is_correct = name in prediction
            
            if is_correct:
                correct_count += 1
            
            results.append({
                "image": img_file,
                "ground_truth": name,
                "prediction": prediction,
                "is_correct": is_correct
            })
            total_images += 1

    # 결과 보고
    accuracy = (correct_count / total_images) * 100 if total_images > 0 else 0
    
    print("\n" + "="*50)
    print(f"TEST RESULTS SUMMARY")
    print(f"Total Images: {total_images}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*50)

    # 상세 결과 저장
    report_path = os.path.join(BASE_DIR, "logs/test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total_images,
                "correct": correct_count,
                "accuracy": accuracy
            },
            "details": results
        }, f, indent=4, ensure_ascii=False)
    
    print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
