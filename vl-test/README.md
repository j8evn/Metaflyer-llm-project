# Vision-Language Model Fine-tuning Project

이 프로젝트는 **Qwen3-VL-30B** 모델을 Fine-tuning하여 특정 인물(아이유, 유재석, 제니 등)을 식별하고 설명하는 Vision-Language Model을 구축하는 것을 목표로 합니다.

## 프로젝트 개요
- **Base Model**: `Qwen/Qwen3-VL-30B-Thinking`
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Hardware**: NVIDIA H100 (80GB)
- **Serving**: vLLM (High-throughput Serving Engine)

## 디렉토리 구조
```bash
vl-test/
├── data/                  # 학습 및 테스트 데이터
│   ├── images/            # 학습용 이미지 (fetch_images.py 실행하면 생성)
│   ├── test_images/       # 테스트용 이미지 (fetch_test_images.py 실행하면 생성)
│   └── dataset.json       # 학습 데이터셋 메타데이터
├── fetch_images.py        # 네이버 이미지 검색 API로 학습 데이터 수집
├── fetch_test_images.py   # 테스트용 이미지 별도 수집
├── train_vl_full.py       # H100용 학습 스크립트 (LoRA, bfloat16)
├── merge_lora.py          # 학습된 LoRA 어댑터를 Base 모델과 병합
├── test_api.py            # vLLM API 서버 테스트 클라이언트
└── README.md              # 프로젝트 설명서
```

## 실행 가이드 (Workflow)

### 1. 데이터 수집 (Data Collection)
네이버 검색 API를 사용하여 인물 이미지를 수집하고 `dataset.json`을 생성합니다.
```bash
python fetch_images.py
```
*   `data/images` 폴더에 이미지가 저장됩니다.
*   `dataset.json`에 이미지 경로와 질문("이 인물은 누구입니까?"), 정답(이름)이 저장됩니다.

### 2. 모델 학습 (Training)
H100 GPU를 사용하여 모델을 학습합니다. (LoRA 적용)
```bash
export CUDA_VISIBLE_DEVICES=2
python train_vl_full.py
```
*   학습 결과는 `output_full/` 디렉토리에 저장됩니다.

### 3. 모델 병합 (Merging)
학습된 LoRA 어댑터(`output_full/checkpoint-XXX`)를 Base 모델과 합쳐서 하나의 모델로 만듭니다.
```bash
python merge_lora.py
```
*   병합된 모델은 `merged_model/` 디렉토리에 저장됩니다.

### 4. 서버 실행 (Serving with vLLM)
병합된 모델을 vLLM을 사용하여 API 서버로 띄웁니다.
```bash
# 기존 vLLM 종료 (필요시)
pkill -f vllm

# 2번 GPU에 서버 실행 (메모리 점유율 90%)
export CUDA_VISIBLE_DEVICES=2
nohup /home/jerry/llm-serve/venv/bin/vllm serve ./merged_model \
--host 0.0.0.0 \
--port 8100 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.9 \
--max_model_len 32768 \
--trust-remote-code \
> vllm.log 2>&1 &
```
*   로그 확인: `tail -f vllm.log`

### 5. 테스트 (Testing)
API를 호출하여 모델이 인물을 잘 맞추는지 테스트합니다.
```bash
# 기본 테스트 (test.jpg)
python test_api.py data/test.jpg

# 특정 이미지 테스트
python test_api.py data/test_images/아이유/test_아이유_002.jpg
```

## 주의사항 및 팁
1.  **OOM (Out of Memory) 에러**:
    *   vLLM과 학습 스크립트를 동시에 같은 GPU에서 돌리면 메모리가 부족합니다.
    *   `nvidia-smi`로 비어있는 GPU를 확인하고 `CUDA_VISIBLE_DEVICES`로 지정해서 사용하세요.
2.  **할루시네이션 (Hallucination)**:
    *   Thinking 모델 특성상 이름 외에 불필요한 설명을 지어낼 수 있습니다.
    *   `test_api.py`의 프롬프트를 "이름만 대답하세요"로 수정하여 제어할 수 있습니다.
3.  **OCR 간섭**:
    *   이미지에 글자(자막, 뉴스 헤드라인)가 있으면 얼굴보다 글자를 먼저 읽을 수 있습니다.
    *   테스트 시 글자가 없는 깨끗한 사진을 사용하는 것이 좋습니다.
