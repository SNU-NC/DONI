import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import os
from typing import Optional, Tuple

class MarketDataCollector:
    """시장 데이터를 수집하는 클래스"""
    
    def __init__(self):
        self.fred = Fred(api_key="a1c8666aaf9b9c79e1a4bd077956ef3e")
    
    def get_risk_free_rate(self) -> float:
        """현재 무위험수익률 조회 (미국 3년 국채)"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 3)
            
            treasury_data = self.fred.get_series('DGS3', 
                                               start_date=start_date,
                                               end_date=end_date)
            
            return float(treasury_data.dropna().iloc[-1]) / 100
            
        except Exception as e:
            print(f"무위험수익률 조회 중 오류 발생: {str(e)}")
            return 0.035
    
    def get_market_risk_premium(self) -> float:
        """시장위험프리미엄 계산
        S&P 500 기준, 3년 기간 동안 미국 국채 3년물 대비 초과 수익률 계산
        (코스피, 한국 국채로 계산하는 경우 마이너스 값 나와서 대체사용)
        시장이 글로벌하게 연결되어 있어서 미국 시장 기준으로 계산해도 무방
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 3)
            
            market_return = self._calculate_market_return(start_date, end_date)
            historical_rf = self._get_historical_risk_free_rate(start_date)
            
            market_risk_premium = market_return - historical_rf
            
            if not (0 <= market_risk_premium <= 0.2):
                return 0.06
                
            return market_risk_premium
            
        except Exception as e:
            print(f"시장위험프리미엄 계산 중 오류 발생: {str(e)}")
            return 0.06
    
    def _calculate_market_return(self, start_date: datetime, end_date: datetime) -> float:
        """특정 기간의 시장 수익률 계산"""
        sp500_start = yf.download('^GSPC',
                                start=start_date,
                                end=start_date + timedelta(days=5),
                                interval='1d',
                                progress=False)
        
        sp500_end = yf.download('^GSPC',
                              start=end_date - timedelta(days=5),
                              end=end_date,
                              interval='1d',
                              progress=False)
        
        start_price = float(sp500_start['Adj Close'].iloc[-1].iloc[0])
        end_price = float(sp500_end['Adj Close'].iloc[-1].iloc[0])
        
        return (end_price / start_price) ** (1/10) - 1
    
    def _get_historical_risk_free_rate(self, date: datetime) -> float:
        """특정 시점의 무위험수익률 조회"""
        treasury_data = self.fred.get_series('DGS3', 
                                           start_date=date,
                                           end_date=date + timedelta(days=30))
        
        return float(treasury_data.dropna().iloc[0]) / 100 