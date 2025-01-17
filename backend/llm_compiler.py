import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatClovaX
from langchain import hub
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List, Any, Dict, Optional, AsyncGenerator
from uuid import UUID
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import logging

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

class StreamingEventHandler(BaseCallbackHandler):
    def __init__(self):
        self.events = []
        self.current_step = None
        logger.info("🔄 StreamingEventHandler 초기화됨")
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        logger.debug(f"🚀 LLM 시작 - 프롬프트: {prompts[:200]}...")  # 프롬프트가 너무 길 수 있으므로 일부만 출력
        step_info = {
            "type": "llm_start",
            "content": "LLM이 생각하는 중입니다...",
            "show_in_chat": True
        }
        self.events.append(step_info)
        self.current_step = step_info
        logger.info("💭 LLM 시작 이벤트 기록됨")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if hasattr(response, 'generations') and response.generations:
            content = response.generations[0][0].text
        else:
            content = str(response)
        
        logger.debug(f"✅ LLM 완료 - 응답: {content[:200]}...")  # 응답이 너무 길 수 있으므로 일부만 출력
        
        step_info = {
            "type": "llm_end",
            "content": content,
            "show_in_chat": False
        }
        self.events.append(step_info)
        self.current_step = step_info
        logger.info("📝 LLM 종료 이벤트 기록됨")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "알 수 없는 도구")
        logger.debug(f"🔧 도구 시작 - {tool_name}: {input_str[:200]}...")
        
        step_info = {
            "type": "tool_start",
            "content": f"🔧 {tool_name} 도구를 사용하여 정보를 찾고 있습니다...",
            "tool_name": tool_name,
            "show_in_chat": True
        }
        self.events.append(step_info)
        self.current_step = step_info
        logger.info(f"🛠️ {tool_name} 도구 시작 이벤트 기록됨")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        summary = output[:200] + "..." if len(output) > 200 else output
        logger.debug(f"🎯 도구 완료 - 결과: {summary}")
        
        step_info = {
            "type": "tool_end",
            "content": f"🎯 도구 실행 결과: {summary}",
            "full_output": output,
            "show_in_chat": True
        }
        self.events.append(step_info)
        self.current_step = step_info
        logger.info("🏁 도구 종료 이벤트 기록됨")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        logger.debug(f"⛓️ 체인 시작 - 입력: {str(inputs)[:200]}...")
        
        step_info = {
            "type": "chain_start",
            "content": "새로운 단계를 시작합니다...",
            "show_in_chat": False
        }
        self.events.append(step_info)
        self.current_step = step_info
        logger.info("🔗 체인 시작 이벤트 기록됨")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        logger.debug(f"🔗 체인 완료 - 출력: {str(outputs)[:200]}...")
        
        step_info = {
            "type": "chain_end",
            "content": "단계가 완료되었습니다.",
            "show_in_chat": False
        }
        self.events.append(step_info)
        self.current_step = step_info
        logger.info("✨ 체인 종료 이벤트 기록됨")

    def get_current_step(self) -> Optional[Dict[str, Any]]:
        if self.current_step:
            logger.debug(f"현재 단계 반환: {self.current_step['type']}")
        return self.current_step

def initialize_chain():
    # LLM 초기화
    llm_4o = ChatOpenAI(model="gpt-4-0125-preview", api_key=os.getenv("OPENAI_API_KEY"))
    llm_mini = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
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

        if state.get("report_agent_use", True):
            print("report agent를 사용합니다.")
            return END

        if isinstance(messages[-1], AIMessage):
            if replan_count >= 2:
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

    async def astream(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"📥 스트리밍 시작 - 쿼리: {query}")
        event_handler = StreamingEventHandler()
        
        state = {
            "messages": [HumanMessage(content=query)],
            "key_information": [],
        }
        logger.debug(f"초기 상태 설정: {state}")
        
        try:
            async for chunk in self.chain.astream(
                state,
                config={"callbacks": [event_handler]}
            ):
                logger.debug(f"청크 수신: {str(chunk)[:200]}...")
                
                current_step = event_handler.get_current_step()
                if current_step and current_step.get("show_in_chat"):
                    logger.info(f"스트리밍 이벤트 전송: {current_step['type']}")
                    yield {
                        "type": "step",
                        "content": current_step["content"],
                        "step_type": current_step["type"]
                    }
                
                if isinstance(chunk, dict) and "messages" in chunk:
                    final_message = chunk["messages"][-1]
                    if isinstance(final_message, AIMessage):
                        logger.info("최종 응답 전송")
                        yield {
                            "type": "final",
                            "content": final_message.content,
                            "events": event_handler.events
                        }
        except Exception as e:
            logger.error(f"스트리밍 중 오류 발생: {str(e)}")
            raise

    def run(self, query: str) -> dict:
        logger.info(f"📝 실행 시작 - 쿼리: {query}")
        try:
            event_handler = StreamingEventHandler()
            
            state = {
                "messages": [HumanMessage(content=query)],
                "key_information": [],
            }
            logger.debug(f"초기 상태 설정: {state}")
            
            result = self.chain.invoke(state, config={"callbacks": [event_handler]})
            logger.debug(f"체인 실행 결과: {str(result)[:200]}...")
            
            final_message = result["messages"][-1]
            
            if isinstance(final_message, AIMessage):
                docs = []
                if "key_information" in result:
                    for info in result["key_information"]:
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
                
                logger.info("✅ 실행 완료 - 성공")
                return {
                    "answer": final_message.content,
                    "docs": docs if docs else [],
                    "events": event_handler.events
                }
            else:
                logger.error("❌ 실행 실패 - 잘못된 응답 형식")
                return {
                    "status": "error",
                    "message": "응답 생성에 실패했습니다.",
                    "events": event_handler.events
                }
                    
        except Exception as e:
            logger.error(f"❌ 실행 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": f"오류가 발생했습니다: {str(e)}",
                "events": event_handler.events if 'event_handler' in locals() else []
            } 