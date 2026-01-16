# VidBrick-VL v1.0 (Qwen3-VL-30B Optimized)

본 프로젝트는 **한국녹색기후기술원(KGcT)**의 성능 품질 시험 기준인 **F1 Score 83% 이상**을 달성하기 위해 최적화된 Qwen3-VL-30B 기반 영상 메타데이터 분석 시스템입니다.

## 주요 구성 파일 설명

1. **`train.py`**:
   - 데이터셋에 최적화된 LoRA(Low-Rank Adaptation) 파인튜닝 스크립트입니다.
   - MoE(Mixture of Experts) 모델의 메모리 효율을 위해 4-bit 양자화 및 Gradient Checkpointing을 적용했습니다.
   - 4,000건의 AI-HUB 한국적 영상 이해 데이터를 사용하여 학습을 진행합니다.

2. **`analytics.py`**:
   - 학습된 모델 또는 원본 모델을 사용하여 영상에서 키워드(대/중/소분류)를 추출합니다.
   - vLLM 서버 없이도 GPU 1, 2, 3번에 모델을 분산 로드하여 안정적으로 추론을 수행합니다.

3. **`f1score.py`**:
   - 추출된 키워드와 정답(Ground Truth) 라벨을 비교하여 Precision, Recall, F1 Score를 산출합니다.

---

## 실행 가이드

### 1. 환경 준비
서버의 CUDA 및 의존성 라이브러리가 올바른지 확인합니다.
```bash
pip install -r requirements.txt
```

### 2. 성능 시험 (Inference & Evaluation)
전체 평가 프로세스를 자동 실행합니다. (H100 GPU 1장 기준 약 30~50분 소요)
```bash
cd /dataset/cep/src
bash run_test.sh
```

### 3. 모델 학습 (Fine-tuning)
목표 성능(83%) 달성을 위해 파인튜닝을 진행합니다. 멀티 GPU 환경에서 안정적으로 돌아가도록 구성되어 있습니다.
```bash
# tmux 세션 사용 권장
tmux new -s qwen_train

# GPU 1, 2, 3번(총 240GB)을 활용하여 학습 시작
export CUDA_VISIBLE_DEVICES=1,2,3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py
```

---

## 주요 기술 및 환경 최적화 내역

*   **인프라 (Storage)**: 모델 가중치 및 캐시 데이터를 100TB 규모의 `/dataset` 파티션에 저장하여 루트 파티션 용량 부족 에러를 원천 차단했습니다.
*   **추론 방식 (Direct Inference)**: 서버의 CUDA 12.2 버전과 vLLM 간의 툴체인 호환성 에러를 우회하기 위해 `Transformers` 정식 라이브러리를 통한 직접 추론 방식을 채택하여 안정성을 100% 확보했습니다.
*   **메모리 관리**: H100 GPU의 HBM3 메모리를 효율적으로 쓰기 위해 BitsAndBytes 4-bit 양자화와 `paged_adamw_32bit` 옵티마이저를 적용했습니다.
*   **정확도 로직**: AI-HUB 데이터셋 특유의 계층 구조(Large/Medium/Small Category)를 모델이 이해할 수 있도록 프롬프트 엔지니어링 및 데이터 콜레이터를 최적화했습니다.

## 문제 해결 (Troubleshooting)

- **CUDA Out of Memory**: GPU 0번은 서비스용으로 점유되어 있을 수 있으므로 반드시 `export CUDA_VISIBLE_DEVICES=1,2,3`과 같이 비어있는 장치를 지정하세요.
- **Syntax Error**: 쉘 스크립트 실행 시 구문 에러가 발생하면 `bash run_test.sh`로 직접 실행해 주세요.
