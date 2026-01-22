#!/bin/bash

# vLLM 서버 백그라운드 실행 스크립트
# Port: 18001
# 실행 방법: chmod +x start_vllm.sh && ./start_vllm.sh

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH="/dataset/cep/llm-training/person-test/merged_model"
LOG_FILE="$BASE_DIR/logs/vllm_server.log"
PYTHON_BIN="$BASE_DIR/venv/bin/python3"

mkdir -p "$BASE_DIR/logs"

echo "Starting vLLM server on port 18001..."
echo "Using virtual environment: $PYTHON_BIN"
echo "Logging to: $LOG_FILE"

# 혹시 이미 18001 포트를 쓰고 있는 프로세스가 있으면 종료
fuser -k 18001/tcp > /dev/null 2>&1

# 가상환경의 python을 사용하여 실행 (CUDA 도구 체인 오류 방지)
# 만약 vllm이 없으면 설치 시도 (처음 한 번만)
if ! "$PYTHON_BIN" -c "import vllm" > /dev/null 2>&1; then
    echo "vLLM not found in venv. Attempting to install..."
    "$PYTHON_BIN" -m pip install vllm==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu121
fi

# 가상환경의 python으로 vLLM 실행
# MoE 모델의 FlashAttention 커널 충돌 방지를 위해 XFORMERS나 TRITON 백엔드 시도 고려 가능
# 여기서는 일단 기본 설정을 유지하되, 에러 발생 시 --attention-backend 설정을 참고바람
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=1 nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 18001 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-model-len 32768 \
  > "$LOG_FILE" 2>&1 &

echo "vLLM server started in background with PID: $!"
echo "You can check logs with: tail -f $LOG_FILE"
