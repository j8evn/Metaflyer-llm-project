import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# 설정
DATA_DIR = "data/images"
REJECTED_DIR = "data/rejected"
REF_DIR = "data/reference"
DATASET_JSON = "data/dataset.json"
CHECK_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SIMILARITY_THRESHOLD = 0.6 # 이 값보다 거리가 멀면 거절 (0~2 사이, 낮을수록 비슷)

def setup_directories():
    if not os.path.exists(REJECTED_DIR):
        os.makedirs(REJECTED_DIR)
    
    # 거절 사유별 폴더 생성
    for reason in ["face_count_mismatch", "identity_mismatch", "error"]:
        path = os.path.join(REJECTED_DIR, reason)
        if not os.path.exists(path):
            os.makedirs(path)

    if not os.path.exists(REF_DIR):
        os.makedirs(REF_DIR)
        print(f"Created reference directory: {REF_DIR}")
        print("Please put reference images (e.g. '아이유_ref.jpg') in this folder.")

def get_embedding(resnet, image_bgr):
    """
    OpenCV(BGR) 이미지를 받아서 FaceNet 임베딩(Tensor)을 반환
    """
    try:
        # BGR -> RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to 160x160 (FaceNet Input)
        img_pil = img_pil.resize((160, 160))
        
        # To Tensor
        img_tensor = transforms.functional.to_tensor(img_pil)
        
        # Standardize (Whiten) - FaceNet typically expects this or specific normalization
        # facenet-pytorch's fixed_image_standardization approximates this
        mean, std = img_tensor.mean(), img_tensor.std()
        img_tensor = (img_tensor - mean) / std
        
        # Add batch dimension and run model
        with torch.no_grad():
            embedding = resnet(img_tensor.unsqueeze(0))
            
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def load_reference_embeddings(resnet):
    embeddings = {}
    if not os.path.exists(REF_DIR):
        return embeddings
        
    print("Loading reference images...")
    for file in os.listdir(REF_DIR):
        if os.path.splitext(file)[1].lower() in CHECK_EXTENSIONS:
            # 파일명에서 이름 추출 (예: 아이유.jpg -> 아이유)
            # _ref 같은 접미사는 제거하고 순수 이름만 매칭
            name = os.path.splitext(file)[0].replace("_ref", "")
            
            path = os.path.join(REF_DIR, file)
            img = cv2.imread(path)
            if img is not None:
                emb = get_embedding(resnet, img)
                if emb is not None:
                    embeddings[name] = emb
                    print(f"Loaded reference for: {name}")
    return embeddings

def process_image(image_path, face_cascade, resnet, ref_embeddings):
    """
    1. 얼굴 감지 및 크롭
    2. 동일인 검증 (Reference가 있을 경우)
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

        # 3. 얼굴 크롭
        x, y, w, h = faces[0]
        margin_x = int(w * 0.5)
        margin_y = int(h * 0.5)
        
        start_x = max(0, x - margin_x)
        start_y = max(0, y - margin_y)
        end_x = min(width, x + w + margin_x)
        end_y = min(height, y + h + margin_y)
        
        cropped_face = image_np[start_y:end_y, start_x:end_x]
        
        # 4. 동일인 검증 (Identity Verification)
        # 파일명에서 이름 추출 (예: data/images/아이유/아이유_001.jpg -> 아이유)
        # 폴더명이 이름이라고 가정, 혹은 파일명 앞부분
        person_name = os.path.basename(os.path.dirname(image_path))
        # 만약 폴더명이 없거나 파일명으로 해야한다면:
        if not person_name or person_name == "images":
             person_name = os.path.basename(image_path).split('_')[0]

        if person_name in ref_embeddings:
            current_emb = get_embedding(resnet, cropped_face)
            ref_emb = ref_embeddings[person_name]
            
            if current_emb is not None:
                # Euclidean Distance
                dist = (current_emb - ref_emb).norm().item()
                
                # print(f"{os.path.basename(image_path)} distance: {dist:.4f}")
                
                if dist > SIMILARITY_THRESHOLD: # 거리가 멀면 다른 사람
                    print(f"Mismatch: {os.path.basename(image_path)} (Dist: {dist:.2f})")
                    return False, "identity_mismatch"

        # 5. 저장
        cv2.imwrite(image_path, cropped_face)
        return True, "cropped"

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

def main():
    print("Initializing Neural Networks...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    setup_directories()
    ref_embeddings = load_reference_embeddings(resnet)
    
    if not ref_embeddings:
        print("Note: No reference images found in data/reference.")
        print("Identity verification will be skipped.")
    
    image_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in CHECK_EXTENSIONS:
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images. Starting cleanup...")
    
    processed_count = 0
    success_count = 0
    rejected_count = 0
    
    for image_path in tqdm(image_files):
        success, reason = process_image(image_path, face_cascade, resnet, ref_embeddings)
        
        if success:
            success_count += 1
        else:
            file_name = os.path.basename(image_path)
            dest_dir = os.path.join(REJECTED_DIR, reason)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.move(image_path, dest_path)
            rejected_count += 1
        
        processed_count += 1

    print("-" * 30)
    print(f"Total: {processed_count}, Success: {success_count}, Rejected: {rejected_count}")
    validate_and_update_dataset()

if __name__ == "__main__":
    main()
