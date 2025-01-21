import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatClovaX
from langchain import hub
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List, Any, Dict, Optional, AsyncGenerator
from uuid import UUID
from langchain_core.messages import AIMessage, FunctionMessage
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import logging
import aiohttp
from datetime import datetime
import asyncio

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 도구 임포트 및 초기화 코드
from tools.webSearch.webSearch_tool import WebSearchTools
from tools.retrieve.analystReport.report_RAG_Tool import ReportRAGTool
from tools.retrieve.financialReport.report_statement_tool import CombinedFinancialReportSearchTool
from tools.math.math_tools import get_math_tool
from tools.financialTerm.fin_knowledge_tools import get_fin_tool
from tools.marketData.market_agent import MarketDataTool
from tools.analyze.stockprice.SameSectorCompareTool import SameSectorAnalyzerTool
from tools.analyze.stockprice.StockAnalyzerTool import StockAnalyzerTool
from tools.analyze.stockprice.CombinedAnalysisTool import CombinedAnalysisTool
from langgraph.graph.message import add_messages

from plan.planner_KB import Planner
from plan.join import create_joiner
from plan.reference import TaskResult, add_task_results

def initialize_chain():
    # LLM 초기화
    llm_4o = ChatOpenAI(model="gpt-4-0125-preview", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    llm_mini = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    clovaX = ChatClovaX(model="HCX-003", clovastudio_api_key=os.getenv("CLOVA_API_KEY"), temperature=0.1)

    # 도구 초기화
    tools = [
        WebSearchTools(llm_mini),
        get_math_tool(llm_4o),
        ReportRAGTool(),
        StockAnalyzerTool(),
        CombinedAnalysisTool(),
        SameSectorAnalyzerTool(),
        CombinedFinancialReportSearchTool(),
        MarketDataTool()
    ]

    # 프롬프트와 플래너 초기화
    prompt = hub.pull("snunc/rag-llm-compiler")
    planner = Planner(llm_4o, llm_mini, clovaX, llm_4o, tools, prompt)
    joiner = create_joiner(llm_4o)  

    # 그래프 상태 정의
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        replan_count: int
        task_results: Annotated[List[TaskResult], add_task_results]
        key_information: List[str]
        report_agent_use: bool
        output_list: List[str]
    # 그래프 생성 및 설정
    graph_builder = StateGraph(State)
    graph_builder.add_node("plan_and_schedule", planner.create_plan_and_schedule)
    graph_builder.add_node("join", joiner)
    graph_builder.add_edge("plan_and_schedule", "join")

    def should_continue(state):
        messages = state["messages"]
        replan_count = state["replan_count"]
        if state.get("report_agent_use", False):
            print("report agent를 사용합니다.")
            return END
        
        print("should_continue의 messages[-1] 확인: ", messages[-1])
        print("should_continue의 replan_count 확인: ", replan_count)
        print("should_continue의 messages 확인: ", messages)
        if isinstance(messages[-1], AIMessage):
            print("messages[-1] 확인: ", messages[-1])
            if replan_count >= 1:
                print("replan_count 이거 때문에 죽음   : ", replan_count)
            return END
        return "plan_and_schedule"

    graph_builder.add_conditional_edges("join", should_continue)
    graph_builder.add_edge(START, "plan_and_schedule")

    return graph_builder.compile() 


class LLMCompiler:
    def __init__(self):
        logger.info("🔨 LLMCompiler 초기화 중...")
        self.chain = initialize_chain()
        logger.info("✅ LLMCompiler 초기화 완료")

    async def arun(self, query: str) -> dict:
        """비동기 실행 메서드"""
        logger.info(f"📝 비동기 실행 시작 - 쿼리: {query}")
        try:
            state = {
                "messages": [HumanMessage(content=query)],
                "key_information": [],
            }
            logger.debug(f"초기 상태 설정: {state}")
            
            current_tasks = []
            final_result = {
                "answer": "",
                "docs": []
            }
            
            # 비동기로 체인 실행
            async for step in self.chain.astream(state):
                logger.debug(f"체인 실행 단계: {str(step)}")
                
                # 계획 단계 처리
                if isinstance(step, dict) and "plan_and_schedule" in step:
                    plan_data = {
                        "type": "plan",
                        "timestamp": datetime.now().isoformat(),
                        "status": "planning",
                        "plan": [
                            {
                                "tool": str(task.get("tool")),
                                "description": str(task.get("args", {})),
                                "status": "pending"
                            }
                            for task in current_tasks
                        ]
                    }
                    await self._update_plan_status(plan_data)
                
                # 태스크 실행 단계 처리
                if isinstance(step, dict) and "join" in step:
                    for msg in step["join"].get("messages", []):
                        if isinstance(msg, FunctionMessage):
                            execution_data = {
                                "type": "execution",
                                "timestamp": datetime.now().isoformat(),
                                "status": "running",
                                "tool": msg.name,
                                "task_id": msg.additional_kwargs.get("idx"),
                                "args": msg.additional_kwargs.get("args", {})
                            }
                            await self._update_execution_status(execution_data)
                
                # 최종 결과 처리
                if isinstance(step, dict) and "join" in step and "messages" in step["join"]:
                    final_message = step["join"]["messages"][-1]
                    if isinstance(final_message, AIMessage):
                        docs = []
                        if "key_information" in step["join"]:
                            for info in step["join"]["key_information"]:
                                doc = {
                                    "tool": info.get("tool", ""),
                                    "referenced_content": info.get("referenced_content", ""),
                                    "filename": info.get("filename"),
                                    "page_number": info.get("page_number"),
                                    "link": info.get("link"),
                                    "title": info.get("title"),
                                    "broker": info.get("broker"),
                                    "target_price": info.get("target_price"),
                                    "investment_opinion": info.get("investment_opinion"),
                                    "analysis_result": info.get("analysis_result", ""),
                                    "content": info.get("content", "")
                                }
                                filtered_doc = {k: v for k, v in doc.items() if v not in [None, ""]}
                                if filtered_doc:
                                    docs.append(filtered_doc)
                        
                        final_result = {
                            "answer": final_message.content,
                            "docs": docs
                        }
            
            logger.info("✅ 비동기 실행 완료 - 성공")
            return final_result
                    
        except Exception as e:
            logger.error(f"❌ 비동기 실행 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": f"오류가 발생했습니다: {str(e)}",
            }

    async def _update_plan_status(self, plan: dict):
        """계획 상태 업데이트"""
        try:
            # plan 데이터가 이미 올바른 형식인지 확인
            plan_data = plan if isinstance(plan, list) else plan.get("plan", [])
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:8000/api/task-progress',
                    json={
                        "type": "plan",
                        "timestamp": datetime.now().isoformat(),
                        "status": "planning",
                        "plan": [
                            {
                                "tool": str(task.get("tool", "")),
                                "description": str(task.get("description", task.get("args", ""))),
                                "status": task.get("status", "pending")
                            }
                            for task in plan_data
                        ]
                    }
                ) as response:
                    if response.status != 200:
                        logger.error("계획 상태 업데이트 실패")
        except Exception as e:
            logger.error(f"계획 상태 업데이트 중 오류: {str(e)}")

    async def _update_execution_status(self, execution: dict):
        """실행 상태 업데이트"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:8000/api/task-progress',
                    json={
                        "type": "execution",
                        "timestamp": datetime.now().isoformat(),
                        "status": "running",
                        "execution": execution
                    }
                ) as response:
                    if response.status != 200:
                        logger.error("실행 상태 업데이트 실패")
        except Exception as e:
            logger.error(f"실행 상태 업데이트 중 오류: {str(e)}")

    # 기존 메서드들은 유지
    async def astream(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"📥 스트리밍 시작 - 쿼리: {query}")
        
        state = {
            "messages": [HumanMessage(content=query)],
            "key_information": [],
        }
        logger.debug(f"초기 상태 설정: {state}")
        
        try:
            async for chunk in self.chain.astream(
                state,
            ):
                logger.debug(f"청크 수신: {str(chunk)[:200]}...")
                
                if isinstance(chunk, dict) and "messages" in chunk:
                    final_message = chunk["messages"][-1]
                    if isinstance(final_message, AIMessage):
                        logger.info("최종 응답 전송")
                        yield {
                            "type": "final",
                            "content": final_message.content,
                        }
        except Exception as e:
            logger.error(f"스트리밍 중 오류 발생: {str(e)}")
            raise

    def run(self, query: str) -> dict:
        """기존의 동기 실행 메서드는 유지"""
        return asyncio.run(self.arun(query)) 