import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# 설정
DATA_DIR = "data/images"
REJECTED_DIR = "data/rejected"
DATASET_JSON = "data/dataset.json" # 데이터셋 파일 경로
CHECK_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def setup_directories():
    if not os.path.exists(REJECTED_DIR):
        os.makedirs(REJECTED_DIR)
    
    # 거절 사유별 폴더 생성
    for reason in ["face_count_mismatch", "error"]:
        path = os.path.join(REJECTED_DIR, reason)
        if not os.path.exists(path):
            os.makedirs(path)

def process_image(image_path, face_cascade):
    """
    이미지를 처리하여 얼굴 부분만 크롭합니다.
    - 얼굴이 1개일 때: 크롭 후 저장 (True 반환)
    - 얼굴이 없거나 2개 이상일 때: 거절 (False 반환)
    """
    try:
        # 1. 이미지 로드
        image_np = cv2.imread(image_path)
        if image_np is None:
            return False, "error"
        
        height, width = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # 2. 얼굴 인식
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) != 1:
            return False, "face_count_mismatch"

        # 3. 얼굴 크롭 (여유 공간 추가)
        x, y, w, h = faces[0]
        
        # 마진 추가 (얼굴 크기의 50%)
        margin_x = int(w * 0.5)
        margin_y = int(h * 0.5)
        
        # 이미지 경계 넘지 않도록 조정
        start_x = max(0, x - margin_x)
        start_y = max(0, y - margin_y)
        end_x = min(width, x + w + margin_x)
        end_y = min(height, y + h + margin_y)
        
        cropped_face = image_np[start_y:end_y, start_x:end_x]
        
        # 4. 크롭된 이미지 덮어쓰기
        cv2.imwrite(image_path, cropped_face)
        
        return True, "cropped"

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, "error"

def main():
    print("Initializing Face Detector (OpenCV)...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    setup_directories()
    
    # 모든 이미지 파일 탐색
    image_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in CHECK_EXTENSIONS:
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images. Starting cleanup (Face Crop)...")
    
    processed_count = 0
    cropped_count = 0
    rejected_count = 0
    
    for image_path in tqdm(image_files):
        # 이미 처리된 파일(작은 파일)인지 확인하는 로직 추가 가능하지만, 일단 덮어쓰기 수행
        success, reason = process_image(image_path, face_cascade)
        
        if success:
            cropped_count += 1
        else:
            # 거절된 파일 이동
            file_name = os.path.basename(image_path)
            dest_dir = os.path.join(REJECTED_DIR, reason)
            dest_path = os.path.join(dest_dir, file_name)
            
            # 파일 이동
            shutil.move(image_path, dest_path)
            rejected_count += 1
        
        processed_count += 1

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total: {processed_count}")
    print(f"Cropped & Saved: {cropped_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Rejected images moved to: {REJECTED_DIR}")
    
    # 데이터셋 JSON 정리
    validate_and_update_dataset()

def validate_and_update_dataset():
    """
    dataset.json을 읽어서 실제 존재하는 이미지만 남기고 저장합니다.
    """
    import json
    
    print("-" * 30)
    print("Validating dataset.json...")
    
    if not os.path.exists(DATASET_JSON):
        print("dataset.json not found. Skipping.")
        return

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaned_data = []
    removed_count = 0
    
    for item in data:
        # data/images/아이유_001.jpg 같은 경로라고 가정
        image_path = item.get("image")
        
        if image_path and os.path.exists(image_path):
            cleaned_data.append(item)
        else:
            removed_count += 1
            
    # 변경된 내용 저장
    if removed_count > 0:
        with open(DATASET_JSON, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        print(f"Updated {DATASET_JSON}: Removed {removed_count} missing entries.")
        print(f"Remaining entries: {len(cleaned_data)}")
    else:
        print("dataset.json is already up to date.")

if __name__ == "__main__":
    main()
