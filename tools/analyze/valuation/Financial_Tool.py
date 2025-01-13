from typing import Optional, Type, List, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults
from tools.analyze.valuation.tools.market_tools import MarketTools
from tools.analyze.valuation.tools.valuation_tools import PERTool, PBRTool
from tools.analyze.valuation.tools.fundamental_tools import EPSTool, BPSTool
from tools.analyze.valuation.tools.peer_tools import PeerPERTool, PeerPBRTool
from langgraph.graph import StateGraph, MessagesState, START, END
import os

class FinancialAnalysisInput(BaseModel):
    """금융 분석 도구의 입력 스키마"""
    query: str = Field(..., description="분석하고자 하는 금융 관련 질의")
    
class FinancialAnalysisTool(BaseTool):
    """금융 데이터 분석을 위한 도구"""
    
    name: str = "financial_analysis_tool"
    description: str = """
    기업의 재무 데이터를 분석하고 관련 정보를 제공합니다:
    - 주가 정보 조회
    - PER, PBR 등 투자 지표 분석
    - 동종 업계 비교 분석
    - 기업 관련 뉴스 검색
    """
    args_schema: Type[BaseModel] = FinancialAnalysisInput
    return_direct: bool = True
    
    # Pydantic 필드로 선언
    tools: List[BaseTool] = Field(default_factory=list, exclude=True)
    llm: Any = Field(default=None, exclude=True)
    tool_node: Any = Field(default=None, exclude=True)
    app: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        load_dotenv()
        # 기본 도구들 초기화
        self.tools = [
            DuckDuckGoSearchResults(backend="news"),
            MarketTools(),
            PERTool(),
            PeerPERTool(),
            EPSTool(),
            PBRTool(),
            PeerPBRTool()
        ]
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-1106-preview",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ).bind_tools(self.tools)
        
        # 도구 노드 생성
        self.tool_node = ToolNode(self.tools)
        
        # 워크플로우 생성
        self._create_workflow()

    def _should_continue(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "end"

    def _call_model(self, state):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _create_workflow(self):
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        self.app = workflow.compile()

    def _run(
        self, 
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """금융 분석 실행"""
        try:
            messages = [("human", query)]
            result = self.app.invoke({"messages": messages})
            
            # 마지막 메시지의 내용을 반환
            final_message = result["messages"][-1]
            return final_message.content
            
        except Exception as e:
            raise ToolException(
                f"금융 분석 도구 오류 발생: {str(e)}\n"
                f"상세 오류: {type(e).__name__}"
            )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """비동기 금융 분석 실행"""
        return self._run(query, run_manager)

# 사용 예시:
"""
from langchain.agents import initialize_agent, AgentType

# 도구 초기화
financial_tool = FinancialAnalysisTool(openai_api_key="your-key")

# 에이전트 초기화
agent = initialize_agent(
    tools=[financial_tool],
    llm=ChatOpenAI(temperature=0),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 실행
result = agent.run("삼성전자의 PER과 주가를 알려줘")
print(result)
"""