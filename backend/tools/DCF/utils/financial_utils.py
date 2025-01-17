from typing import Optional, Dict
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def calculate_beta(ticker: str, market_index: str = '^KS11', period: int = 730) -> float:
    """베타 계산"""
    try:
        stock_data = yf.download(ticker, 
                               start=(datetime.now() - timedelta(days=period)),
                               end=datetime.now(),
                               interval='1wk')
        
        market_data = yf.download(market_index,
                                start=(datetime.now() - timedelta(days=period)),
                                end=datetime.now(),
                                interval='1wk')
        
        stock_returns = stock_data['Adj Close'].pct_change().dropna()
        market_returns = market_data['Adj Close'].pct_change().dropna()
        
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        beta = covariance / market_variance
        
        if not (0 <= beta <= 3):
            return 1.0
            
        return beta
        
    except Exception:
        return 1.0

def get_yfinance_beta(ticker: str) -> Optional[float]:
    """yfinance에서 베타값 조회"""
    try:
        beta = yf.Ticker(ticker).info.get('beta')
        if beta is not None and 0 <= beta <= 3:
            return beta
        return None
    except Exception:
        return None 
    
def calculate_effective_tax_rate(metrics: Dict[str, float]) -> float:
    """실효세율 계산"""
    try:
        pretax_income = metrics['pretax_income']
        income_tax = metrics['tax_provision']
        
        if pretax_income == 0:
            return 0.22
        
        effective_tax_rate = income_tax / pretax_income
        
        if not (0 <= effective_tax_rate <= 1):
            return 0.22
            
        return effective_tax_rate
        
    except Exception:
        return 0.22