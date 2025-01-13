import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatClovaX
from langchain import hub
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List
from langchain_core.messages import HumanMessage, AIMessage

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

from planner_KB import Planner
from join import create_joiner
from reference import TaskResult, add_task_results

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

        if isinstance(messages[-1], AIMessage):
            if replan_count >= 2:
                print("replan_count 이거 때문에 죽음   : ", replan_count)
            return END
        return "plan_and_schedule"

    graph_builder.add_conditional_edges("join", should_continue)
    graph_builder.add_edge(START, "plan_and_schedule")

    return graph_builder.compile() 