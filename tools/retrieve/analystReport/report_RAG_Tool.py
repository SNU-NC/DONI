from typing import List, Optional, Any, Type, Dict
from langchain_core.tools import BaseTool
import logging
import asyncio
import nest_asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from tools.retrieve.analystReport.retrievers_02 import WebScrapeRetriever
from config.prompts import _Analyst_RAG_DESCRIPTION
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# 중첩된 이벤트 루프 허용
nest_asyncio.apply()

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
    web_scrape_retriever: Optional[WebScrapeRetriever] = Field(default=None, exclude=True)
    logger: logging.Logger = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.web_scrape_retriever = WebScrapeRetriever()
    
    def _format_error_message(self, query: str) -> str:
        return (
            f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.\n"
            "다음과 같은 방법을 시도해보세요:\n"
            "1. 더 구체적인 검색어로 다시 시도\n"
            "2. 유사한 의미의 다른 애널리스트 보고서 관련 검색어 사용"
        )

    def _run(
        self,
        query: str,
        company: str,
        year: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict:
        """동기 실행"""
        try:
            if company is None:
                return {
                    "output": "회사명(company) 인자가 필요합니다. 회사명을 입력해주세요.",
                    "key_information": []
                }
            
            if not hasattr(asyncio, '_get_running_loop') or asyncio._get_running_loop() is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                loop = asyncio.get_event_loop()
            
            future = self.web_scrape_retriever.run(query=query, company=company)
            results = loop.run_until_complete(future)
            
            if not results or not results.get("output"):
                return {
                    "output": self._format_error_message(query),
                    "key_information": []
                }
            
            return results
            
        except Exception as e:
            error_msg = f"검색 에이전트(리포트레그) 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}"
            self.logger.error(error_msg)
            return {
                "output": error_msg,
                "key_information": []
            }
    
    async def _arun(
        self,
        query: str,
        company: str,
        year: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict:
        """비동기 실행"""
        try:
            if company is None:
                return {
                    "output": "회사명(company) 인자가 필요합니다. 회사명을 입력해주세요.",
                    "key_information": [],
                }
            
            results = await self.web_scrape_retriever.run(query=query, company=company)
            
            if not results or not results.get("output"):
                return {
                    "output": self._format_error_message(query),
                    "key_information": []
                }
            
            return results
            
        except Exception as e:
            error_msg = f"검색 에이전트(리포트레그) 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}"
            self.logger.error(error_msg)
            return {
                "output": error_msg,
                "key_information": []
            }