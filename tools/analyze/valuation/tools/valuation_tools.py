from typing import Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from ..utils.ticker_finder import TickerFinder
import yfinance as yf

class ValuationInput(BaseModel):
    company: str = Field(description="회사명 (예: 삼성전자, Apple)")

class ValuationBaseTool(BaseTool):
    def get_ticker_data(self, company: str) -> tuple:
        """회사의 티커 정보와 yfinance Ticker 객체를 반환합니다."""
        ticker_symbol = TickerFinder.get_ticker(company)
        if ticker_symbol is None:
            return None, None
        return ticker_symbol, yf.Ticker(ticker_symbol)

class PERTool(ValuationBaseTool):
    name: str = "per_analysis_tool"
    description: str = "기업의 PER(주가수익비율)을 계산합니다"
    args_schema: type[ValuationInput] = ValuationInput
    
    def _run(self, company: str) -> Optional[dict]:
        _, ticker = self.get_ticker_data(company)
        if ticker is None:
            return None
        
        earning_ttm = sum(ticker.quarterly_income_stmt.loc['Net Income Common Stockholders'][:4])
        per = ticker.info["marketCap"]/earning_ttm
        return {"PER": per}
    
    def _arun(self, company: str):
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")

class PBRTool(ValuationBaseTool):
    name: str = "pbr_analysis_tool"
    description: str = "기업의 PBR(주가순자산비율)을 계산합니다"
    args_schema: type[ValuationInput] = ValuationInput

    
    def _run(self, company: str) -> Optional[dict]:
        _, ticker = self.get_ticker_data(company)
        if ticker is None:
            return None
        
        equity = ticker.quarterly_balance_sheet.loc['Stockholders Equity'][0]
        pbr = ticker.info["marketCap"]/equity
        return {"PBR": pbr}
    
    def _arun(self, company: str):
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")