#!/bin/bash

PORT=8000
MAX_RETRIES=5
RETRY_DELAY=2

cleanup_port() {
    echo "Checking for processes using port $PORT..."
    
    # 포트를 사용중인 모든 프로세스 ID 가져오기
    local pids=$(lsof -ti:$PORT)
    
    if [ ! -z "$pids" ]; then
        echo "Found processes using port $PORT: $pids"
        echo "Attempting to terminate processes..."
        
        # SIGTERM으로 먼저 시도
        kill $pids 2>/dev/null || true
        sleep 1
        
        # 여전히 실행 중인 프로세스 확인
        pids=$(lsof -ti:$PORT)
        if [ ! -z "$pids" ]; then
            echo "Forcing termination of remaining processes..."
            kill -9 $pids 2>/dev/null || true
        fi
    else
        echo "No processes found using port $PORT"
    fi
}

wait_for_port() {
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if ! lsof -i:$PORT >/dev/null 2>&1; then
            echo "Port $PORT is available"
            return 0
        fi
        echo "Port $PORT still in use, waiting..."
        retries=$((retries + 1))
        sleep $RETRY_DELAY
    done
    return 1
}

# 메인 실행 로직
echo "Starting server setup..."

# 포트 정리
cleanup_port

# 포트가 사용 가능할 때까지 대기
if wait_for_port; then
    echo "Starting uvicorn server..."
    cd backend && uvicorn app:app --reload --port $PORT
else
    echo "ERROR: Could not free up port $PORT after $MAX_RETRIES attempts"
    echo "Please check running processes manually with: lsof -i:$PORT"
    exit 1
fi