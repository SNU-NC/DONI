"""
RAG 시스템 메인 모듈
"""

from typing import List, Tuple, Optional, Any, Type, Dict
from langchain_core.tools import BaseTool
from tools.retrieve.financialReport.retrievers import RetrievalManager
from tools.retrieve.financialReport.vector_store import VectorStoreManager
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from config.prompts import _Financial_RAG_DESCRIPTION

class FinancialReportInputSchema(BaseModel):
    query: str = Field(..., description="검색 문장")
    company: str = Field(..., description="검색 회사명")
    year: int = Field(default=2023, description="검색 연도")

class FinancialReportTool(BaseTool):
    """사업보고서 검색을 위한 RAG 시스템 도구"""
    
    name: str = "financial_report_search"
    description: str = _Financial_RAG_DESCRIPTION
    args_schema: Type[BaseModel] = FinancialReportInputSchema
    return_direct: bool = True
    retrieval_manager: RetrievalManager = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        vector_store_manager = VectorStoreManager()
        self.retrieval_manager = RetrievalManager(vector_store_manager.load_or_create(create_flag=False))
    
    def _run(
        self, 
        query: str, 
        company: str = None,
        year: int = 2023,
        top_k: int = 5, 
        rewrite: bool = True, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            if company is None:
                return "회사명(company) 인자가 필요합니다. 회사명을 입력해주세요."
            results = self.retrieval_manager.get_retriever_results(
                query=query, 
                k=top_k, 
                rewrite=rewrite,
                metadata={"companyName": company, "year": year}
            )
            if not results:
                return (
                    f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.\n"
                    "다음과 같은 방법을 시도해보세요:\n"
                    "1. 더 구체적인 검색어로 다시 시도\n"
                    "2. 유사한 의미의 다른 사업보고서 관련 검색어 사용\n"
                    "만약 계속해서 결과가 없다면, 다른 정보 소스나 도구 사용을 고려해보세요."
                )
            
            return results
            
        except Exception as e:
            raise ToolException(f"검색 에이전트 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}")
    
    async def _arun(
        self, 
        query: str, 
        company: str = None,
        year: int = 2023,
        top_k: int = 5, 
        rewrite: bool = False, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        try:
            results = self.retrieval_manager.get_retriever_results(
                query=query, 
                k=top_k, 
                rewrite=rewrite,
                metadata={"companyName": company, "year": year}
            )
            if not results:
                return (
                    f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.\n"
                    "다음과 같은 방법을 시도해보세요:\n"
                    "1. 더 구체적인 검색어로 다시 시도\n"
                    "2. 유사한 의미의 다른 사업보고서 관련 검색어 사용\n"
                    "만약 계속해서 결과가 없다면, 다른 정보 소스나 도구 사용을 고려해보세요."
                )
            
            return results
            
        except Exception as e:
            raise ToolException(f"검색 에이전트 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}")
