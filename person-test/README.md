# Vision-Language Model Fine-tuning Project (Qwen3-VL)

이 프로젝트는 **Qwen3-VL-30B-MoE** 모델을 Fine-tuning하여 특정 인물을 식별하고 설명하는 개인화된 Vision-Language Model을 구축하는 것을 목표로 합니다.

## 디렉토리 구조
```text
person-test/
├── config/             # 설정 파일 모음 (인물 리스트 등)
├── data/               # 데이터셋 모음
│   ├── train/          # 학습용 원본 이미지
│   ├── test/           # 검증용 테스트 이미지
│   └── dataset.json    # 정제된 학습 데이터셋 메타데이터
├── scripts/            # 실행 코드 모음
│   ├── fetch_images.py
│   ├── preprocess_data.py
│   ├── train.py
│   ├── merge_lora.py
│   └── test_api.py
├── models/             # 결과물 모음
│   ├── weights/        # 학습된 LoRA 어댑터 가중치
│   └── merged/         # vLLM용 최종 병합 모델
└── logs/               # 실행 로그 및 vLLM 로그
```

## 파이프라인 실행 가이드

### 1. 설정 및 데이터 수집
`config/people.json`에 학습할 인물 리스트를 작성한 뒤 데이터를 수집합니다. (기본 1인당 20장)
```bash
# 1. 학습용 이미지 수집
python scripts/fetch_images.py

# 2. 테스트용 이미지 수집 (검증용)
python scripts/fetch_test_images.py

# 3. 데이터 정제 및 얼굴 인식/크롭
python scripts/preprocess_data.py
```

### 2. 모델 학습 (Training)
H100 GPU를 사용하여 모델을 학습합니다.
```bash
python scripts/train.py
```
*   학습된 LoRA 어댑터는 `models/weights/`에 저장됩니다.

### 3. 모델 병합 (Merging)
학습된 어댑터를 베이스 모델과 병합하여 배포용 모델을 생성합니다.
```bash
python scripts/merge_lora.py
```
*   최종 모델은 `models/merged/`에 저장됩니다.

### 4. 서버 서빙 및 테스트
vLLM을 사용하여 배포하고 API 요청을 보냅니다.
```bash
# 1. vLLM 서버 기동 (안격적인 V0 엔진 사용 권장, 백그라운드 실행)
export CUDA_VISIBLE_DEVICES=1  # 사용할 GPU 번호
nohup env VLLM_USE_V1=0 vllm serve ./models/merged \
    --port 8100 \
    --gpu_memory_utilization 0.9 \
    --trust-remote-code \
    > logs/vllm.log 2>&1 &

# 2. 인물 식별 테스트
python scripts/test_api.py data/test/아이유/test_아이유_000.jpg
```

## 💡 주요 팁
1. **정교한 관리**: 인물 추가를 원하시면 `config/people.json`만 수정하면 모든 스크립트에 자동 반영됩니다.
2. **배포 환경**: vLLM 0.14.0 사용 시 `VLLM_USE_V1=0` 환경 변수를 사용해야 MoE 모델 로딩 버그를 피할 수 있습니다.
3. **데이터 보안**: 대용량 이미지와 모델 가중치는 Git에 업로드되지 않도록 `.gitignore` 설정이 되어 있습니다.
