import json
import os

EXTRACTED_FILE = "extracted_metadata.json"
LABEL_DIR = "/dataset/cep/37.한국적_영상_이해_데이터/3.개방데이터/1.데이터/Validation_Sample_1000/02.라벨링데이터"

def analyze_errors():
    if not os.path.exists(EXTRACTED_FILE):
        print("extracted_metadata.json이 없습니다.")
        return
        
    with open(EXTRACTED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat1_hit, cat2_hit, cat3_hit = 0, 0, 0
    total = 0

    for filename, pred in data.items():
        label_path = os.path.join(LABEL_DIR, filename + ".json")
        if not os.path.exists(label_path): continue
        
        with open(label_path, "r", encoding="utf-8") as f:
            gt_content = json.load(f)
            gt = gt_content.get("image", {})
        
        pk = pred.get("keywords", [])
        gk = [gt.get(f"image_category_{i}", "").strip().replace("'", "").replace('"', "") for i in range(1, 4)]
        
        # 모델 출력값 정제
        pk_clean = [k.strip().replace("'", "").replace('"', "") for k in pk]
        
        if len(pk_clean) >= 1 and len(gk) >= 1 and pk_clean[0] == gk[0]: cat1_hit += 1
        if len(pk_clean) >= 2 and len(gk) >= 2 and pk_clean[1] == gk[1]: cat2_hit += 1
        if len(pk_clean) >= 3 and len(gk) >= 3 and pk_clean[2] == gk[2]: cat3_hit += 1
        total += 1

    if total == 0:
        print("분석할 데이터가 없습니다.")
        return

    print("=" * 40)
    print(f"분석 결과 (총 {total}건)")
    print("-" * 40)
    print(f"1단계(대분류) 일치: {cat1_hit}건 ({cat1_hit/total*100:.2f}%)")
    print(f"2단계(중분류) 일치: {cat2_hit}건 ({cat2_hit/total*100:.2f}%)")
    print(f"3단계(소분류) 일치: {cat3_hit}건 ({cat3_hit/total*100:.2f}%)")
    print("=" * 40)

if __name__ == "__main__":
    analyze_errors()
