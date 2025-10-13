#!/bin/bash

# API 서버 시작 스크립트

echo "======================================"
echo "LLM API 서버 시작"
echo "======================================"

# 환경 변수 로드
if [ -f .env ]; then
    echo "환경 변수 로딩 중..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# 기본값 설정
MODEL_PATH=${MODEL_PATH:-"outputs/checkpoints/final_model"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}

echo ""
echo "설정:"
echo "  모델 경로: $MODEL_PATH"
echo "  호스트: $HOST"
echo "  포트: $PORT"
echo ""

# 모델 경로 확인
if [ ! -d "$MODEL_PATH" ]; then
    echo "오류: 모델 경로를 찾을 수 없습니다: $MODEL_PATH"
    echo "MODEL_PATH 환경 변수를 올바르게 설정하세요."
    exit 1
fi

echo "API 서버 시작 중..."
echo ""
echo "문서 확인: http://localhost:$PORT/docs"
echo "종료하려면 Ctrl+C를 누르세요"
echo ""

# API 서버 실행
python src/api_server.py \
    --model_path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    ${LOAD_IN_8BIT:+--load_in_8bit} \
    ${LOAD_IN_4BIT:+--load_in_4bit}

