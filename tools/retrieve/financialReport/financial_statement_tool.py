"""
재무제표 테이블 분석 도구 모듈
"""

from typing import List, Tuple, Optional, Any, Type, Dict
from langchain_core.tools import BaseTool
from tools.retrieve.financialReport.retrievers import RetrievalManager
from tools.retrieve.financialReport.vector_store import VectorStoreManager
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from config.prompts import _Financial_TABLE_DESCRIPTION
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import OpenDartReader
from tools.retrieve.financialReport.prompts import TABLE_AGENT_PREFIX
from tools.retrieve.financialReport.utils import preprocess_financial_df

class FinancialStatementInputSchema(BaseModel):
    query: str = Field(..., description="재무제표 검색을 위한 질문")
    company: str = Field(..., description="회사명")
    year: int = Field(..., description="연도")
    class Config:
        extra = "allow"  # 추가 필드 허용

class FinancialStatementTool(BaseTool):
    """재무제표 검색 도구"""
    
    name: str = "financial_statement_search"
    description: str = _Financial_TABLE_DESCRIPTION
    args_schema: Type[BaseModel] = FinancialStatementInputSchema
    return_direct: bool = True
    dart: Any = Field(default=None, exclude=True)
    llm: Any = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self.dart = OpenDartReader("4925a6e6e69d8f9138f4d9814f56f371b2b2079a")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
    
    def _get_financial_data(self, companyName: str, year: int) -> Tuple[pd.DataFrame, Optional[str]]:
        """DART에서 해당 연도의 재무제표 데이터 가져오기"""
        try:
            print(f"입력 회사명 : {companyName}, 입력 연도 : {year}")
            
            # 3년치 데이터 가져오기
            dfs = []
            latest_rcp_no = None
            
            for y in range(year, year-3, -1):
                df = self.dart.finstate_all(companyName, y, reprt_code="11011")
                if df is None or df.empty:
                    df = self.dart.finstate(companyName, y, reprt_code="11011")
                if df is not None and not df.empty:
                    dfs.append(df)
                    if latest_rcp_no is None:
                        latest_rcp_no = df['rcept_no'].iloc[-1]
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                return combined_df, latest_rcp_no
            
        except Exception as e:
            print(f"재무제표 조회 실패: {str(e)}, 입력 회사명 : {companyName}, 입력 연도 : {year}")
        return pd.DataFrame(), None

    def _create_financial_agent(self, df: pd.DataFrame) -> Any:
        """재무제표 분석을 위한 Pandas 에이전트 생성"""
        if df.empty:
            return None
            
        return create_pandas_dataframe_agent(
            self.llm,
            df,
            prefix=TABLE_AGENT_PREFIX,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            allow_dangerous_code=True,
        )

    def _run(
        self, 
        query: str,
        company: str,
        year: int = 2023,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        try:
            # 회사명 확인
            if not company:
                return {
                    "output": "회사명이 제공되지 않았습니다. 분석하고자 하는 회사명을 입력해주세요.",
                    "key_information": []
                }
            
            # 회사 정보 구성
            company_info = {
                "companyName": company,
                "year": year
            }
            
            # 재무제표 데이터와 접수번호 가져오기
            df, latest_rcp_no = self._get_financial_data(company_info["companyName"], company_info["year"])
            print("사업 보고서 번호: ", latest_rcp_no)
            if df.empty:
                return {
                    "output": f"'{company_info['companyName']}'의 {company_info['year']}년도 재무제표를 찾을 수 없습니다.",
                    "key_information": []
                }
            df = preprocess_financial_df(df)
            # 재무제표 분석 에이전트 생성
            print(df.head())
            agent = self._create_financial_agent(df)
            if not agent:
                return {
                    "output": "재무제표 분석 에이전트를 생성할 수 없습니다.",
                    "key_information": []
                }
            
            # 분석 수행
            analysis_result = agent.invoke({"input": query})
            analysis_result = analysis_result['output']
            # 문서 URL 가져오기 (접수번호가 있는 경우 직접 조회)
            doc_url = None
            if latest_rcp_no:
                try:
                    docs = self.dart.sub_docs(str(latest_rcp_no), "연결 재무제표")
                    if docs is not None and not docs.empty:
                        doc_url = docs['url'].iloc[0]
                except Exception as e:
                    print(f"문서 URL 조회 실패: {str(e)}")
            
            return {
                "output": analysis_result,
                "key_information": [{
                    "tool": "재무제표 도구",
                    "referenced_content": analysis_result,
                    "page_number": "연결 재무제표",
                    "filename": f"{company_info['companyName']}_{company_info['year']}_financial_statement",
                    "link": doc_url if doc_url else "https://dart.fss.or.kr/"
                }]
            }
            
        except Exception as e:
            raise ToolException(f"재무제표 분석 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}")
    
    async def _arun(
        self,
        query: str,
        company: str,
        year: int = 2023,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        return self._run(query, company, year, run_manager) 