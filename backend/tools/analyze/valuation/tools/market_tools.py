from typing import Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from ..utils.ticker_finder import TickerFinder
import yfinance as yf

class MarketInput(BaseModel):
    company: str = Field(description="회사명 (예: 삼성전자, Apple)")

class MarketTools(BaseTool):
    name: str = "stock_price_tool"
    description: str = "기업의 주가 정보를 조회합니다"
    args_schema: type[MarketInput] = MarketInput
    
    def get_ticker_data(self, company: str) -> tuple:
        """회사의 티커 정보와 yfinance Ticker 객체를 반환합니다."""
        ticker_symbol = TickerFinder.get_ticker(company)
        if ticker_symbol is None:
            return None, None
        return ticker_symbol, yf.Ticker(ticker_symbol)
    
    def _run(self, company: str) -> Optional[dict]:
        """기업의 어제 종가를 찾습니다."""
        _, ticker = self.get_ticker_data(company)
        if ticker is None:
            return None
        return {"어제 종가": ticker.info["regularMarketPreviousClose"]}
        
    def _arun(self, company: str):
        """비동기 실행은 지원하지 않습니다."""
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")