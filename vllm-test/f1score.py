import os
import json
import glob

# 설정
VALIDATION_BASE_DIR = "/dataset/cep/37.한국적_영상_이해_데이터/3.개방데이터/1.데이터/Validation_Sample_1000"
LABEL_DIR = os.path.join(VALIDATION_BASE_DIR, "02.라벨링데이터")
EXTRACTED_FILE = "extracted_metadata.json"
RESULT_FILE = "result.json"

def calculate_f1():
    print("=" * 60)
    print("F1 Score 평가 시작")
    print("=" * 60)

    if not os.path.exists(EXTRACTED_FILE):
        print(f"error: {EXTRACTED_FILE} 파일을 찾을 수 없습니다. analytics.py를 먼저 실행하세요.")
        return

    with open(EXTRACTED_FILE, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    processed_count = 0

    for filename, data in extracted_data.items():
        # 정답(GT) 로드
        label_path = os.path.join(LABEL_DIR, filename + ".json")
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            gt_content = json.load(f)

        # GT 키워드 추출 및 정제
        img_info = gt_content.get("image", {})
        gt_keywords = []
        for i in range(1, 4):
            cat = img_info.get(f"image_category_{i}")
            if cat and cat.strip():
                # 따옴표 및 공백 제거
                clean_cat = cat.strip().replace("'", "").replace('"', "")
                gt_keywords.append(clean_cat)
        gt_keywords = set(gt_keywords)

        # 예측(Pred) 키워드 정제
        pred_keywords = set([k.strip().replace("'", "").replace('"', "") for k in data.get("keywords", [])])

        # TP, FP, FN 계산
        tp = len(pred_keywords.intersection(gt_keywords))
        fp = len(pred_keywords - gt_keywords)
        fn = len(gt_keywords - pred_keywords)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        processed_count += 1

    # 최종 Precision, Recall, F1
    precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0
    recall = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    result = {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "total_samples": processed_count,
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(json.dumps(result, indent=4))
    print("평가 완료!")
    print("=" * 60)

if __name__ == "__main__":
    calculate_f1()
