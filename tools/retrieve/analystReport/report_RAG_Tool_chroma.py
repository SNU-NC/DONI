"""
애널리스트 리포트 RAG 시스템 메인 모듈
"""

from typing import List, Optional, Any, Type, Dict
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.retrieve.analystReport.vector_store_02 import VectorStoreManager
from tools.retrieve.analystReport.retrievers_02_chroma import RetrievalManager
from langchain_core.tools import ToolException
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from config.prompts import _Analyst_RAG_DESCRIPTION

class ReportRAGInputSchema(BaseModel):
    query: str = Field(..., description="검색 문장")
    company: str = Field(..., description="검색 회사명")
    year: int = Field(default=2024, description="검색 연도")

class ReportRAGTool(BaseTool):
    """재무 데이터 검색을 위한 RAG 시스템 도구"""
    
    name: str = "report_rag_search"
    description: str = _Analyst_RAG_DESCRIPTION
    args_schema: Type[BaseModel] = ReportRAGInputSchema
    return_direct: bool = True
    retrieval_manager: RetrievalManager = Field(default=None, exclude=True)
    logger: logging.Logger = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        vector_store_manager = VectorStoreManager()
        # load_or_create()의 반환값에서 vectorstore만 추출
        vectorstore, _ = vector_store_manager.load_or_create(create_flag=False)
        self.retrieval_manager = RetrievalManager(vectorstore)
    
    def _run(self, query: str, company: str, year: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            if company is None:
                return "회사명(company) 인자가 필요합니다. 회사명을 입력해주세요."
            
            # 검색 필터 설정
            search_filter = {
                "$and": [
                    {"companyName": company},
                    {"year": year}
                ]
            }
                
            # 검색 실행
            results = self.retrieval_manager.get_retriever_results(
                query=query.strip('"').strip(),
                filter=search_filter,
                k=4
            )
            
            if not results:
                return (
                    f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.\n"
                    "다음과 같은 방법을 시도해보세요:\n"
                    "1. 더 구체적인 검색어로 다시 시도\n"
                    "2. 유사한 의미의 다른 애널리스트 보고서 관련 검색어 사용"
                )
            
            return results['combined_content']
            
        except Exception as e:
            error_msg = f"검색 에이전트(리포트레그) 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}"
            self.logger.error(error_msg)
            raise ToolException(error_msg)
    
    async def _arun(
        self,
        query: str,
        company: str,
        year: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """비동기 실행은 동기 실행과 동일한 로직 사용"""
        return self._run(query=query, company=company, year=year, run_manager=run_manager)
