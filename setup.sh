#!/bin/bash

# LLM 파인튜닝 프로젝트 설정 스크립트

echo "========================================"
echo "LLM 파인튜닝 프로젝트 설정"
echo "========================================"

# 가상환경 생성
echo ""
echo "[1/5] 가상환경 생성 중..."
python3 -m venv venv

# 가상환경 활성화
echo ""
echo "[2/5] 가상환경 활성화..."
source venv/bin/activate

# pip 업그레이드
echo ""
echo "[3/5] pip 업그레이드 중..."
pip install --upgrade pip

# 의존성 설치
echo ""
echo "[4/5] 의존성 패키지 설치 중..."
pip install -r requirements.txt

# 디렉토리 생성 및 샘플 데이터 생성
echo ""
echo "[5/5] 샘플 데이터 생성 중..."
python src/data_utils.py

echo ""
echo "========================================"
echo "설정 완료!"
echo "========================================"
echo ""
echo "사용 방법:"
echo "  1. 가상환경 활성화: source venv/bin/activate"
echo "  2. 학습 실행: python src/train.py --config configs/train_config.yaml"
echo "  3. 추론 실행: python src/inference.py --model_path models/your-model"
echo ""
echo "자세한 내용은 README.md를 참조하세요."

