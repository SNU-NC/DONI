from typing import Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from ..utils.ticker_finder import TickerFinder
import yfinance as yf

class FundamentalInput(BaseModel):
    company: str = Field(description="회사명 (예: 삼성전자, Apple)")

class FundamentalBaseTool(BaseTool):
    def get_ticker_data(self, company: str) -> tuple:
        """회사의 티커 정보와 yfinance Ticker 객체를 반환합니다."""
        ticker_symbol = TickerFinder.get_ticker(company)
        if ticker_symbol is None:
            return None, None
        return ticker_symbol, yf.Ticker(ticker_symbol)

class EPSTool(FundamentalBaseTool):
    name: str = "eps_analysis_tool"
    description: str = "기업의 EPS(주당순이익)을 계산합니다"
    args_schema: type[FundamentalInput] = FundamentalInput
    
    def _run(self, company: str) -> Optional[dict]:
        _, ticker = self.get_ticker_data(company)
        if ticker is None:
            return None
            
        earning_ttm = sum(ticker.quarterly_income_stmt.loc['Net Income Common Stockholders'][:4])
        eps = earning_ttm/ticker.info["sharesOutstanding"]
        return {"EPS": eps}
    
    def _arun(self, company: str):
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")

class BPSTool(FundamentalBaseTool):
    name: str = "bps_analysis_tool"
    description: str = "기업의 BPS(주당순자산가치)를 계산합니다"
    args_schema: type[FundamentalInput] = FundamentalInput
    
    def _run(self, company: str) -> Optional[dict]:
        _, ticker = self.get_ticker_data(company)
        if ticker is None:
            return None
            
        equity = ticker.quarterly_balance_sheet.loc['Stockholders Equity'][0]
        bps = equity/ticker.info["sharesOutstanding"]
        return {"BPS": bps}
    
    def _arun(self, company: str):
        raise NotImplementedError("비동기 실행은 지원하지 않습니다") 