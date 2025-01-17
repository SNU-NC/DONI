from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import signal
import atexit
import pandas as pd
import ast
import logging
from pathlib import Path
import json
from typing import List, Dict, Any
import asyncio

# uvloop 비활성화
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# 기존 RAG 관련 임포트 
from llm_compiler import LLMCompiler

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프로젝트 루트 경로 확인
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend")

# frontend 디렉토리 존재 확인
if not os.path.exists(FRONTEND_PATH):
    os.makedirs(FRONTEND_PATH)
    print(f"Created frontend directory at: {FRONTEND_PATH}")

# 정적 파일 서빙 설정
app.mount("/frontend", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
app.mount("/component", StaticFiles(directory=os.path.join(FRONTEND_PATH, "component")), name="component")

# LLMCompiler 인스턴스 생성
llm_compiler = LLMCompiler()

class Query(BaseModel):
    query: str

class ChatMessage(BaseModel):
    role: str
    content: str

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            query = json.loads(data)["query"]
            
            # 스트리밍 응답 시작
            async for step in llm_compiler.astream(query):
                await websocket.send_json(step)
                
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "content": f"오류가 발생했습니다: {str(e)}"
        })
    finally:
        await websocket.close()

@app.post("/chat")
async def chat(query: str):
    return llm_compiler.run(query)

@app.post("/api/search")
async def search(query: Query):
    try:
        result = llm_compiler.run(query.query)
        
        if isinstance(result, dict) and result.get("status") == "error":
            return {"error": True, "message": result["message"]}
            
        return {
            "error": False,
            "answer": result["answer"] if isinstance(result, dict) else str(result),
            "docs": result.get("docs", []) if isinstance(result, dict) else []
        }
        
    except Exception as e:
        logging.error(f"Search API 오류: {str(e)}")
        return {"error": True, "message": "서버 오류가 발생했습니다."}

# 메인 페이지 리다이렉트
@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

# 서버 종료 시 정리 함수
def cleanup():
    print("\nServer shutting down...")
    # 8000 포트 사용중인 프로세스 종료
    try:
        os.system("kill -9 $(lsof -t -i:8000)")
    except:
        pass

# 종료 시그널 핸들러
def signal_handler(signum, frame):
    cleanup()
    exit(0)

# 종료 시그널 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 프로그램 종료 시 cleanup 함수 실행
atexit.register(cleanup)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, loop="asyncio")