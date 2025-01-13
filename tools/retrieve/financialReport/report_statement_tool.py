"""
RAG 시스템과 재무제표 분석을 결합한 통합 검색 도구 모듈
"""

from typing import List, Tuple, Optional, Any, Type, Dict, Literal
from langchain_core.tools import BaseTool
from tools.retrieve.financialReport.retrievers import RetrievalManager
from tools.retrieve.financialReport.vector_store import VectorStoreManager
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from config.prompts import _COMBINED_FINANCIAL_REPORT_DESCRIPTION
from tools.retrieve.financialReport.prompts import COMBINED_FINANCIAL_REPORT_PROMPT
from tools.retrieve.financialReport.financial_report_tool import FinancialReportTool
from tools.retrieve.financialReport.financial_statement_tool import FinancialStatementTool
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, ToolMessage
import json

class CombinedInputSchema(BaseModel):
    query: str = Field(..., description="검색 문장")
    company: str = Field(..., description="검색 회사명")
    year: int = Field(default=2023, description="검색 연도")


class CombinedFinancialReportSearchTool(BaseTool):
    """사업보고서와 재무제표를 통합 검색하는 도구"""
    
    name: str = "combined_financial_report_search"
    description: str = _COMBINED_FINANCIAL_REPORT_DESCRIPTION
    args_schema: Type[BaseModel] = CombinedInputSchema
    return_direct: bool = True
    
    report_tool: FinancialReportTool = Field(default=None, exclude=True)
    statement_tool: FinancialStatementTool = Field(default=None, exclude=True)
    tool_node: ToolNode = Field(default=None, exclude=True)
    llm: ChatOpenAI = Field(default=None, exclude=True)
    llm_with_tools: ChatOpenAI = Field(default=None, exclude=True)
    workflow: StateGraph = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self.report_tool = FinancialReportTool()
        self.statement_tool = FinancialStatementTool()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        # 도구 목록 설정
        tools = [self.report_tool, self.statement_tool]
        self.tool_node = ToolNode(tools, handle_tool_errors=True)


        self.llm_with_tools = self.llm.bind_tools(tools)
            
        # 워크플로우 설정
        self._setup_workflow()

    def _setup_workflow(self):
        """워크플로우 설정"""
        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return END

        def call_model(state: MessagesState):
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        
        self.workflow = workflow.compile()

    def _run(
        self, 
        query: str, 
        company: str = None,
        year: int = 2023,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        try:
            if company is None:
                return {"error": "회사명(company) 인자가 필요합니다. 회사명을 입력해주세요."}
            # 워크플로우 실행
            messages = [
                (
                    "system",
                    COMBINED_FINANCIAL_REPORT_PROMPT
                ),
                (
                    "human",
                    f"""
                    회사: {company}
                    연도: {year}
                    사용자 질문: {query}
                    """
                )
            ]
            
            result = self.workflow.invoke({
                "messages": messages
            })
            
            # 결과 처리
            final_message = result["messages"][-1]
            key_information = []
            # 도구 호출 결과에서 key_information 수집
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    try:
                        content = json.loads(msg.content)
                        if "key_information" in content:
                            key_information.extend(content["key_information"])
                    except:
                        pass
            return {
                "output": final_message.content,
                "key_information": key_information
            }
            
        except Exception as e:
            raise ToolException(f"통합 검색 도구 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}")
    
    async def _arun(
        self, 
        query: str, 
        company: str = None,
        year: int = 2023,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        return self._run(query, company, year, run_manager)
