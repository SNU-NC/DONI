

from tools.analyze.stockprice.CompanyAnalyzerTool import CompanyAnalyzerTool
from tools.analyze.stockprice.StockAnalyzerTool import StockAnalyzerTool
from tools.analyze.stockprice.SameSectorCompareTool import SameSectorAnalyzerTool
from tools.analyze.valuation.Financial_Tool import FinancialAnalysisTool
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from typing import List,Optional, Any, Type
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLLM
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from config.prompts import _ANALYZER_DESCRIPTION

value_tool = CompanyAnalyzerTool()
compare_tool = SameSectorAnalyzerTool()
explain_tool = StockAnalyzerTool()
valuation_tool = FinancialAnalysisTool()



class AnlaAgentInputSchema(BaseModel):
    query: str = Field(..., description="검색 하고자 하는 문장")


class CombinedStockAnalyzerTool(BaseTool):
    """재무 데이터 검색을 위한 RAG 시스템 도구"""
    
    name: str = "CombinedCompanyAnalyzer"
    description: str =_ANALYZER_DESCRIPTION
    args_schema: Type[BaseModel] = AnlaAgentInputSchema
    value_tool : BaseTool = Field(default=None, exclude=True)
    compare_tool : BaseTool = Field(default=None, exclude=True)
    explain_tool : BaseTool = Field(default=None, exclude=True)
    valuation_tool : BaseTool = Field(default=None, exclude=True)
    llm : BaseLLM = Field(default=None, exclude=True)

 
    prompt: str = Field(default=None, exclude=True)

    def __init__(self, llm : BaseLLM):
        super().__init__()
        self.value_tool = CompanyAnalyzerTool()
        self.compare_tool = SameSectorAnalyzerTool()
        self.explain_tool = StockAnalyzerTool()
        self.valuation_tool = FinancialAnalysisTool()
        self.llm = llm
        _AGENT_PROMPT= """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of tools
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
Thought:{agent_scratchpad}'''
"""
        self.prompt=PromptTemplate.from_template(_AGENT_PROMPT)


    def _run(self, query: str) -> str:
        try:
            analyze_manager = create_react_agent(self.llm, tools=[self.value_tool, self.compare_tool, self.explain_tool,self.valuation_tool], state_modifier=_ANALYZER_DESCRIPTION)
            for s in analyze_manager.stream({"messages": [("user", query)]}):
                if isinstance(s, tuple):
                    print(s)
            results = s
            if not results:
                return (
                    f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.\n"
                    "다음과 같은 방법을 시도해보세요:\n"
                    "1. 더 구체적인 검색어로 다시 시도\n"
                    "2. 코스피에 상장된 기업인지 확인해 보세요\n"
                    "만약 계속해서 결과가 없다면, 다른 정보 소스나 도구 사용을 고려해보세요."
                )
            return f"분석 결과 : {results}"

            #return "분석 결과: " + results['agent']['messages'][0].content            
        except Exception as e:
            raise ToolException(f"분석 에이전트 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}")
    
    async def _arun(self, query: str) -> str:
        try:
            analyze_manager = create_react_agent(self.llm, tools=[self.value_tool, self.compare_tool, self.explain_tool], state_modifier=_ANALYZER_DESCRIPTION)
            for s in analyze_manager.stream({"messages": [("user", query)]}):
                if isinstance(s, tuple):
                    print(s)
            results = s
            if not results:
                return (
                    f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.\n"
                    "다음과 같은 방법을 시도해보세요:\n"
                    "1. 더 구체적인 검색어로 다시 시도\n"
                    "2. 코스피에 상장된 기업인지 확인해 보세요\n"
                    "만약 계속해서 결과가 없다면, 다른 정보 소스나 도구 사용을 고려해보세요."
                )
            return f"분석 결과 : {results}"
            #return "분석 결과: " + results['agent']['messages'][0].content              
        except Exception as e:
            raise ToolException(f"분석 에이전트 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}")
    