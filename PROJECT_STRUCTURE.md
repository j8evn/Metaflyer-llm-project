# 프로젝트 구조

```
llm/
├── README.md                    # 프로젝트 개요 및 설명서
├── QUICKSTART.md               # 빠른 시작 가이드
├── EXAMPLES.md                 # 상세 사용 예제
├── PROJECT_STRUCTURE.md        # 이 파일 - 프로젝트 구조 설명
├── requirements.txt            # Python 의존성 패키지
├── setup.sh                    # 자동 설정 스크립트
├── .gitignore                  # Git 무시 파일 목록
│
├── configs/                    # 설정 파일 디렉토리
│   └── train_config.yaml       # 학습 설정 (모델, 데이터, 하이퍼파라미터 등)
│
├── src/                        # 소스 코드 디렉토리
│   ├── __init__.py            # 패키지 초기화
│   ├── train.py               # 메인 학습 스크립트
│   ├── inference.py           # 추론/예측 스크립트
│   ├── model_utils.py         # 모델 관련 유틸리티 (로딩, LoRA 적용 등)
│   └── data_utils.py          # 데이터 처리 유틸리티 (로딩, 전처리 등)
│
├── scripts/                    # 유틸리티 스크립트
│   ├── create_sample_data.py  # 샘플 데이터 생성
│   ├── evaluate_model.py      # 모델 평가
│   ├── convert_checkpoint.py  # LoRA 가중치 병합
│   └── quantize_model.py      # 모델 양자화
│
├── data/                       # 데이터 디렉토리
│   ├── .gitkeep               # Git 디렉토리 유지용
│   ├── train.json             # 학습 데이터 (예시)
│   └── eval.json              # 평가 데이터 (예시)
│
├── models/                     # 모델 저장 디렉토리
│   └── .gitkeep               # Git 디렉토리 유지용
│
├── outputs/                    # 학습 출력 디렉토리
│   └── .gitkeep               # Git 디렉토리 유지용
│   ├── checkpoints/           # 학습 체크포인트 (자동 생성)
│   └── logs/                  # 학습 로그 (자동 생성)
│
└── notebooks/                  # Jupyter 노트북
    └── quickstart.ipynb       # 빠른 시작 노트북
```

## 주요 파일 설명

### 설정 파일

#### `configs/train_config.yaml`
학습에 필요한 모든 설정을 포함:
- 모델 설정 (이름, 경로)
- 데이터 설정 (경로, 최대 길이)
- 학습 설정 (에포크, 배치 크기, 학습률)
- LoRA 설정 (rank, alpha, target modules)
- 양자화 설정 (4bit/8bit)
- 모니터링 설정 (WandB, TensorBoard)

### 소스 코드

#### `src/train.py`
메인 학습 스크립트:
- 설정 로딩 및 파싱
- 모델 및 토크나이저 초기화
- 데이터셋 로딩 및 전처리
- Trainer 설정 및 학습 실행
- 모델 저장

사용법:
```bash
python src/train.py --config configs/train_config.yaml
```

#### `src/inference.py`
추론 스크립트:
- 학습된 모델 로딩
- 텍스트 생성
- 대화형 모드 지원
- Instruction 형식 지원

사용법:
```bash
python src/inference.py --model_path "모델경로"
```

#### `src/model_utils.py`
모델 관련 유틸리티:
- `ModelLoader`: 모델 로딩 클래스
- LoRA 설정 및 적용
- 양자화 설정
- 파라미터 정보 출력

#### `src/data_utils.py`
데이터 처리 유틸리티:
- `DatasetLoader`: 데이터셋 로딩 클래스
- JSON 파일 로딩
- Instruction/Chat/Text 형식 지원
- 토크나이징
- 데이터셋 분할

### 스크립트

#### `scripts/create_sample_data.py`
샘플 데이터 생성:
- Instruction 형식
- Chat 형식
- Text 형식

사용법:
```bash
python scripts/create_sample_data.py --format all
```

#### `scripts/evaluate_model.py`
모델 평가:
- 평가 데이터셋으로 모델 테스트
- 결과를 JSON으로 저장

사용법:
```bash
python scripts/evaluate_model.py \
    --model_path "모델경로" \
    --eval_data "data/eval.json"
```

#### `scripts/convert_checkpoint.py`
LoRA 병합:
- LoRA 어댑터를 베이스 모델과 병합
- 단일 모델로 저장

사용법:
```bash
python scripts/convert_checkpoint.py \
    --base_model "베이스모델" \
    --lora_model "LoRA경로" \
    --output "출력경로"
```

#### `scripts/quantize_model.py`
모델 양자화:
- 4bit 또는 8bit 양자화
- 모델 크기 축소

사용법:
```bash
python scripts/quantize_model.py \
    --model_path "모델경로" \
    --bits 8
```

## 데이터 형식

### Instruction 형식
```json
[
    {
        "instruction": "질문 또는 지시사항",
        "input": "추가 입력 (선택)",
        "output": "기대되는 출력"
    }
]
```

### Chat 형식
```json
[
    {
        "messages": [
            {"role": "user", "content": "사용자 메시지"},
            {"role": "assistant", "content": "어시스턴트 응답"}
        ]
    }
]
```

### Text 형식
```json
[
    {
        "text": "학습할 텍스트"
    }
]
```

## 워크플로우

### 1. 환경 설정
```bash
./setup.sh
# 또는 수동으로
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# 샘플 데이터 생성
python scripts/create_sample_data.py --format all

# 또는 자신의 데이터 준비
# data/train.json, data/eval.json
```

### 3. 설정 조정
```bash
# configs/train_config.yaml 편집
vim configs/train_config.yaml
```

### 4. 학습 실행
```bash
python src/train.py --config configs/train_config.yaml
```

### 5. 추론 테스트
```bash
python src/inference.py --model_path "outputs/checkpoints/final_model"
```

### 6. 평가 (선택)
```bash
python scripts/evaluate_model.py \
    --model_path "outputs/checkpoints/final_model" \
    --eval_data "data/eval.json"
```

## 확장 가능성

이 프로젝트 구조는 다음과 같이 확장할 수 있습니다:

1. **새로운 데이터 형식 추가**
   - `src/data_utils.py`에 새로운 포맷 함수 추가

2. **커스텀 학습 전략**
   - `src/train.py`에 커스텀 Trainer 클래스 구현

3. **추가 평가 메트릭**
   - `scripts/evaluate_model.py` 확장

4. **배포 스크립트**
   - `scripts/`에 배포 관련 스크립트 추가

5. **웹 인터페이스**
   - Gradio/Streamlit 등으로 UI 추가

자세한 내용은 각 파일의 docstring과 주석을 참조하세요!

