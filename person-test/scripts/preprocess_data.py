import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

# 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/train")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data/test")
REJECTED_DIR = os.path.join(BASE_DIR, "data/rejected")
DATASET_JSON = os.path.join(BASE_DIR, "data/dataset.json")
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
    1. 얼굴 감지 및 크롭
    2. 얼굴이 여러 개면 Reject (데이터 순도 유지)
    3. 얼굴이 0개면 원본 사용 (Too many removed 방지용 Fallback)
    """
    try:
        image_np = cv2.imread(image_path)
        if image_np is None:
            return False, "error"
        
        height, width = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # 얼굴 인식 실패 시: 일단 원본 사용 (데이터 확보 우선)
            cropped_face = image_np
            reason = "no_face_fallback"
        elif len(faces) > 1:
            # 얼굴이 여러 개면: 누가 주인공인지 알 수 없으므로 Reject
            return False, "face_count_mismatch"
        else:
            # 얼굴이 딱 하나면: 크롭 진행
            x, y, w, h = faces[0]
            margin_x = int(w * 0.5)
            margin_y = int(h * 0.5)
            
            start_x = max(0, x - margin_x)
            start_y = max(0, y - margin_y)
            end_x = min(width, x + w + margin_x)
            end_y = min(height, y + h + margin_y)
            
            cropped_face = image_np[start_y:end_y, start_x:end_x]
            reason = "cropped"

        cv2.imwrite(image_path, cropped_face)
        return True, reason

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, "error"

def validate_and_update_dataset():
    import json
    print("-" * 30)
    print("Validating dataset.json...")
    if not os.path.exists(DATASET_JSON):
        return
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned_data = []
    removed_count = 0
    for item in data:
        if item.get("image") and os.path.exists(item.get("image")):
            cleaned_data.append(item)
        else:
            removed_count += 1
    if removed_count > 0:
        with open(DATASET_JSON, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        print(f"Updated dataset.json: Removed {removed_count} entries.")

def process_directory(directory, face_cascade):
    print(f"Scanning directory: {directory}...")
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in CHECK_EXTENSIONS:
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No images found in {directory}.")
        return

    print(f"Found {len(image_files)} images in {directory}. Processing...")
    
    processed_count = 0
    success_count = 0
    rejected_count = 0
    
    for image_path in tqdm(image_files):
        success, reason = process_image(image_path, face_cascade)
        
        if success:
            success_count += 1
        else:
            file_name = os.path.basename(image_path)
            dest_dir = os.path.join(REJECTED_DIR, reason)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.move(image_path, dest_path)
            rejected_count += 1
        
        processed_count += 1
        
    print(f"[{directory}] Total: {processed_count}, Success: {success_count}, Rejected: {rejected_count}")

def main():
    print("Initializing Face Detection...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    setup_directories()
    
    process_directory(DATA_DIR, face_cascade)
    
    if os.path.exists(TEST_DATA_DIR):
        print("-" * 30)
        process_directory(TEST_DATA_DIR, face_cascade)
    
    print("-" * 30)
    validate_and_update_dataset()

if __name__ == "__main__":
    main()
