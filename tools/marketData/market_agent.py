"""
마켓 데이터 분석 에이전트 모듈
"""
from typing import List, Dict, Any, Type, Optional
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from tools.marketData.market_tools import create_market_tools
from tools.marketData.prompts import MARKET_AGENT_PROMPT
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ToolException
from config.prompts import MARKET_Data_Tool_PROMPT

class MarketDataInputSchema(BaseModel):
    query: str = Field(..., description="사용자의 금융 시장 데이터 관련 질의")

class MarketDataTool(BaseTool):
    """금융 시장 데이터 분석 도구"""
    name: str = "market_data_tool"
    description: str = MARKET_Data_Tool_PROMPT
    args_schema: Type[BaseModel] = MarketDataInputSchema
    return_direct: bool = True
    tools: List[BaseTool] = Field(default_factory=list)
    llm: Any = Field(default=None)
    agent_executor: Any = Field(default=None)

    def __init__(self):
        super().__init__()
        self.tools = create_market_tools()
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4"
        )
        
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_modifier=MARKET_AGENT_PROMPT
        )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        사용자 질의를 분석하고 응답을 생성합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            str: 분석 결과
        """
        try:
            result = self.agent_executor.invoke({"messages": [("human", query)]})
            return result["messages"][-1].content
                
        except Exception as e:
            raise ToolException(f"분석 중 오류가 발생했습니다: {str(e)}")

    @property
    def available_tools(self) -> List[BaseTool]:
        """사용 가능한 도구 목록을 반환합니다."""
        return self.tools