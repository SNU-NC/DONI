import yfinance as yf
from ..utils.ticker_finder import TickerFinder

class BaseFinancialTool:
    """금융 도구의 기본 클래스"""
    
    @staticmethod
    def get_ticker_data(company: str) -> tuple:
        """회사의 티커 정보와 yfinance Ticker 객체를 반환합니다."""
        ticker_symbol = TickerFinder.get_ticker(company)
        if ticker_symbol is None:
            return None, None
        return ticker_symbol, yf.Ticker(ticker_symbol) 