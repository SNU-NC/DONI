import logging
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from tools.analyze.report_agent.tools.report_agent import ReportAgentManager

_REPORT_AGENT_DESCRIPTION = """report_agent(query: str , company_name: str) -> str:
    목표주가 계산 및 리포트 작성 에이전트, 사업부별 매출 성장률과 뉴스를 활용하여 목표주가 산정 후 목표주가 산정 프로세스 관련 리포트 작성
"""

class ReportAgentInputSchema(BaseModel):
    query: str = Field(..., description="검색하고자 하는 문장")
    metadata: Optional[Dict[str, Any]] = Field(None, description="메타데이터 필터 (회사명, 연도 등)")

class ReportAgentTool(BaseTool):
    "목표주가 계산 및 리포트 작성 에이전트, 사업부별 매출 성장률과 뉴스를 활용하여 목표주가 산정 후 목표주가 산정 프로세스 관련 리포트 작성"

    name: str = "report_agent"
    description: str = _REPORT_AGENT_DESCRIPTION
    args_schema: Type[BaseModel] = ReportAgentInputSchema
    return_direct: bool = True # 리포트 작성 결과를 유저에게 다이렉트로 반환
    report_agent: ReportAgentManager = Field(default=None, exclude=True)
    logger: logging.Logger = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.report_agent = ReportAgentManager()

    def _run(self, query: str, company_name: str = "삼성전자", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            filter_dict = {}
            filter_dict["companyName"] = company_name

            # 검색 필터 설정
            search_filter = filter_dict if filter_dict else None

            # 리포트 작성실행
            report = self.report_agent.get_report(
                query=query.strip('"').strip(),
                filter=search_filter,
            )
            return report
        except Exception as e:
            return f"Error: {e}"