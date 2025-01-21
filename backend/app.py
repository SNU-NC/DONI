from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, WebSocketDisconnect
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
from datetime import datetime
import uuid

# uvloop 비활성화
#asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# 기존 RAG 관련 임포트 
from llm_compiler import LLMCompiler

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 서버
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

class Query(BaseModel):
    query: str

class ChatMessage(BaseModel):
    role: str
    content: str

# 전역 상태 관리
task_progress = []
current_plans = {}
execution_queue = asyncio.Queue()

# LLMCompiler 인스턴스 생성
llm_compiler = LLMCompiler()

# 도구 타입별 표시 이름 매핑
TOOL_TYPE_NAMES = {
    'WebSearchTools': '웹 검색',
    'MathTool': '수식 계산',
    'ReportRAGTool': '리포트 분석',
    'StockAnalyzerTool': '주가 분석',
    'CombinedAnalysisTool': '종합 분석',
    'SameSectorAnalyzerTool': '섹터 분석',
    'CombinedFinancialReportSearchTool': '재무제표 검색',
    'MarketDataTool': '시장 데이터'
}

async def execute_tasks_background(task_id: str):
    """백그라운드에서 태스크를 실행하는 함수"""
    try:
        plan = current_plans.get(task_id)
        if not plan:
            return
        
        # 실행 상태 업데이트
        task_progress.append({
            "type": "execution_start",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "status": "running"
        })
        
        # 실제 태스크 실행
        result = await llm_compiler.arun(plan["query"])
        
        # 실행 완료 상태 업데이트
        task_progress.append({
            "type": "execution_complete",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "status": "completed",
            "result": result
        })
        
    except Exception as e:
        logging.error(f"Task execution error: {str(e)}")
        task_progress.append({
            "type": "execution_error",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        })

@app.post("/api/plan")
async def create_plan(query: Query, background_tasks: BackgroundTasks):
    """계획 생성 엔드포인트"""
    try:
        # 계획 생성
        task_id = str(uuid.uuid4())
        current_plans[task_id] = {
            "query": query.query,
            "timestamp": datetime.now().isoformat(),
            "status": "planning"
        }
        
        # 계획 상태 업데이트
        task_progress.append({
            "type": "plan",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "status": "created",
            "query": query.query
        })
        
        # 백그라운드에서 실행
        background_tasks.add_task(execute_tasks_background, task_id)
        
        return {
            "task_id": task_id,
            "status": "accepted"
        }
        
    except Exception as e:
        logging.error(f"Plan creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """특정 태스크의 상태를 조회하는 엔드포인트"""
    try:
        # 해당 태스크의 모든 진행 상황 조회
        task_events = [
            event for event in task_progress 
            if event.get("task_id") == task_id
        ]
        
        if not task_events:
            raise HTTPException(status_code=404, detail="Task not found")
            
        return {
            "task_id": task_id,
            "events": task_events,
            "current_status": task_events[-1]["status"]
        }
        
    except Exception as e:
        logging.error(f"Task status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        # 새로운 질문이 들어오면 이전 task_progress 초기화
        task_progress.clear()
        
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

@app.websocket("/ws/task-progress")
async def task_progress_websocket(websocket: WebSocket):
    await websocket.accept()
    last_sent_index = 0
    
    try:
        while True:
            if len(task_progress) > last_sent_index:
                new_events = task_progress[last_sent_index:]
                for event in new_events:
                    # 이벤트 데이터 정리
                    simplified_event = {
                        "type": event.get("type"),
                        "status": event.get("status"),
                        "task_id": event.get("task_id"),
                        "task_name": event.get("task_name"),
                        "timestamp": event.get("timestamp"),
                        "tool_type": TOOL_TYPE_NAMES.get(
                            event.get("tool_type", ""),
                            event.get("tool_type", "기타")
                        )
                    }
                    
                    # 결과 데이터가 있는 경우 추가
                    if event.get("result"):
                        if isinstance(event["result"], dict):
                            simplified_event["result"] = {
                                "status": event["result"].get("status"),
                                "message": event["result"].get("message", ""),
                                "tool_type": TOOL_TYPE_NAMES.get(
                                    event["result"].get("tool_type", ""),
                                    event.get("tool_type", "기타")
                                )
                            }
                        else:
                            simplified_event["result"] = str(event["result"])
                    
                    # 실행 정보가 있는 경우 추가
                    if event.get("execution"):
                        simplified_event.update(event["execution"])
                    
                    try:
                        await websocket.send_json({
                            "type": "task_progress",
                            "data": simplified_event
                        })
                    except WebSocketDisconnect:
                        logging.info("WebSocket 연결이 종료되었습니다.")
                        return
                    except Exception as e:
                        logging.error(f"메시지 전송 중 오류 발생: {e}")
                        continue
                        
                last_sent_index = len(task_progress)
                
            await asyncio.sleep(0.1)  # 폴링 간격 줄임
            
    except WebSocketDisconnect:
        logging.info("WebSocket 연결이 종료되었습니다.")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logging.error(f"WebSocket 종료 중 오류 발생: {e}")

@app.post("/api/task-progress")
async def update_task_progress(data: dict):
    """태스크 진행 상황을 업데이트하는 엔드포인트"""
    try:
        # 도구 타입 정보 추가
        if "tool" in data and isinstance(data["tool"], dict):
            tool_type = data["tool"].get("type", "")
            data["tool_display_name"] = TOOL_TYPE_NAMES.get(tool_type, tool_type)
        
        # 데이터 크기 제한
        if "debug_info" in data:
            data["debug_info"] = {
                k: str(v)[:200] if isinstance(v, str) else v 
                for k, v in data["debug_info"].items()
            }
        
        if "result" in data:
            if isinstance(data["result"], dict):
                data["result"] = {
                    k: str(v)[:500] if isinstance(v, str) else v 
                    for k, v in data["result"].items()
                }
                # 결과에도 도구 타입 정보 추가
                if "tool_type" in data["result"]:
                    data["result"]["tool_display_name"] = TOOL_TYPE_NAMES.get(
                        data["result"]["tool_type"],
                        data["result"]["tool_type"]
                    )
            elif isinstance(data["result"], str):
                data["result"] = data["result"][:500]
        
        required_fields = ["timestamp", "status"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"필수 필드가 누락되었습니다: {field}")
        
        MAX_TASK_PROGRESS = 100
        task_progress.append(data)
        if len(task_progress) > MAX_TASK_PROGRESS:
            task_progress.pop(0)
        
        return {
            "status": "success",
            "message": "태스크 진행 상황이 업데이트되었습니다.",
            "data": data
        }
        
    except Exception as e:
        logging.error(f"태스크 진행 상황 업데이트 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/task-progress")
async def get_task_progress():
    """
    현재까지의 태스크 진행 상황을 조회하는 엔드포인트
    """
    try:
        # 진행 상황을 시간순으로 정렬
        sorted_progress = sorted(task_progress, key=lambda x: x.get("timestamp", ""))
        
        # 타입별로 데이터 구성
        response = {
            "plans": [item for item in sorted_progress if item.get("type") == "plan"],
            "executions": [item for item in sorted_progress if item.get("type") == "execution"],
            "others": [item for item in sorted_progress if item.get("type") not in ["plan", "execution"]]
        }
        
        return response
    except Exception as e:
        logging.error(f"태스크 진행 상황 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, loop="asyncio")