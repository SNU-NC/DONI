#!/bin/bash

# 포트 설정
BACKEND_PORT=8000
FRONTEND_PORT=5173
MAX_RETRIES=5
RETRY_DELAY=2

cleanup_port() {
    local port=$1
    echo "포트 ${port}를 사용하는 프로세스 확인 중..."
    
    local pids=$(lsof -ti:$port)
    
    if [ ! -z "$pids" ]; then
        echo "포트 ${port}를 사용하는 프로세스 발견: $pids"
        echo "프로세스 종료 시도 중..."
        
        kill $pids 2>/dev/null || true
        sleep 1
        
        pids=$(lsof -ti:$port)
        if [ ! -z "$pids" ]; then
            echo "남은 프로세스 강제 종료..."
            kill -9 $pids 2>/dev/null || true
        fi
    else
        echo "포트 ${port}를 사용하는 프로세스가 없습니다"
    fi
}

wait_for_port() {
    local port=$1
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if ! lsof -i:$port >/dev/null 2>&1; then
            echo "포트 $port 사용 가능"
            return 0
        fi
        echo "포트 $port가 아직 사용 중, 대기 중..."
        retries=$((retries + 1))
        sleep $RETRY_DELAY
    done
    return 1
}

# 메인 실행 로직
echo "서버 설정 시작..."

# 포트 정리
cleanup_port $BACKEND_PORT
cleanup_port $FRONTEND_PORT

# 포트가 사용 가능할 때까지 대기
if wait_for_port $BACKEND_PORT && wait_for_port $FRONTEND_PORT; then
    echo "백엔드와 프론트엔드 서버 시작 중..."
    
    # 백엔드 서버 시작 (백그라운드로)
    cd backend && uvicorn app:app --reload --port $BACKEND_PORT &
    BACKEND_PID=$!
    
    # 프론트엔드 서버 시작 (백그라운드로)
    cd ../frontend && npm run dev &
    FRONTEND_PID=$!
    
    # 프로세스 ID 저장
    echo $BACKEND_PID > .backend.pid
    echo $FRONTEND_PID > .frontend.pid
    
    echo "백엔드 서버가 http://localhost:${BACKEND_PORT} 에서 실행 중"
    echo "프론트엔드 서버가 http://localhost:${FRONTEND_PORT} 에서 실행 중"
    echo "서버를 종료하려면 Ctrl+C를 누르세요"
    
    # 종료 시그널 처리
    trap 'cleanup_port $BACKEND_PORT; cleanup_port $FRONTEND_PORT; exit 0' SIGINT SIGTERM
    
    # 백그라운드 프로세스가 실행 중인 동안 대기
    wait
else
    echo "에러: $MAX_RETRIES번의 시도 후에도 포트를 사용할 수 없습니다"
    echo "실행 중인 프로세스를 확인해주세요:"
    echo "백엔드 포트: lsof -i:$BACKEND_PORT"
    echo "프론트엔드 포트: lsof -i:$FRONTEND_PORT"
    exit 1
fi 