import os
import shutil
import glob
import random

# 설정
SOURCE_BASE = "/dataset/cep/37.한국적_영상_이해_데이터/3.개방데이터/1.데이터/Validation"
TARGET_BASE = "/dataset/cep/37.한국적_영상_이해_데이터/3.개방데이터/1.데이터/Validation_Sample_1000"

SRC_IMG = os.path.join(SOURCE_BASE, "01.원천데이터")
SRC_LBL = os.path.join(SOURCE_BASE, "02.라벨링데이터")

TGT_IMG = os.path.join(TARGET_BASE, "01.원천데이터")
TGT_LBL = os.path.join(TARGET_BASE, "02.라벨링데이터")

def sample_data(count=1000):
    # 디렉토리 생성
    os.makedirs(TGT_IMG, exist_ok=True)
    os.makedirs(TGT_LBL, exist_ok=True)

    # 라벨 기준 파일 목록 확보
    label_files = sorted(glob.glob(os.path.join(SRC_LBL, "*.json")))
    print(f"전체 파일 개수: {len(label_files)}")

    # 랜덤 샘플링 (재현성을 위해 시드 고정 가능)
    # random.seed(42) 
    sampled_labels = random.sample(label_files, min(count, len(label_files)))

    print(f"{len(sampled_labels)}개 데이터 복사 시작...")

    for lbl_path in sampled_labels:
        file_name = os.path.basename(lbl_path)
        img_name = file_name.replace(".json", ".png")
        img_path = os.path.join(SRC_IMG, img_name)

        if os.path.exists(img_path):
            # 파일 복사
            shutil.copy2(lbl_path, os.path.join(TGT_LBL, file_name))
            shutil.copy2(img_path, os.path.join(TGT_IMG, img_name))
        else:
            print(f"경고: 이미지를 찾을 수 없음 - {img_name}")

    print(f"완료! 저장 위치: {TARGET_BASE}")

if __name__ == "__main__":
    sample_data(1000)
