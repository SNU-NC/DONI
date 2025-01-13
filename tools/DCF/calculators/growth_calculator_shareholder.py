from typing import Dict, Optional
from tools.DCF.collectors.financial_data_collector import FinancialDataCollector
import pandas as pd
from tools.DCF.collectors.info_data_collector import InfoDataCollector

class GrowthCalculatorShareholder:
    """순이익 성장률을 계산하는 클래스"""
    
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.financial_collector = FinancialDataCollector(ticker_symbol)
        self.info_collector = InfoDataCollector(ticker_symbol)

    def calculate_net_income_growth_rate(self, period: str = "annual", years: int = 1) -> Dict[str,float]:
        """순이익 성장률 계산 (순이익 성장률 = 유보율(1-배당률) * ROE(Net Income / Equity))
        
        Args:
            period (str): 기간 (annual, quarterly)
            years (int): 평균을 계산할 연도 수
        Returns:
            Dict[str, float]: 성장률 계산 결과

            Growth Rate (float): 순이익 성장률
            Retention Ratio (float): 유보율
                    : 순이익에서 배당금을 제외하고 투자를 위해 남긴 돈
                ROE (float): 자기자본수익률
                    : 주주가 투자한 돈으로 얼마나 효율적으로 이익을 냈는지 보여주는 지표.
        """
        try:
            # 재무 지표 수집
            metrics = self.financial_collector.extract_financial_metrics(period)
            info = self.info_collector.get_info()
            
            # n년 평균 ROE 계산
            roe_values = []
            for i in range(min(years, len(metrics['net_income']))):
                current_roe = metrics['net_income'].iloc[i] / metrics['total_equity'].iloc[i]
                roe_values.append(current_roe)
            
            roe = sum(roe_values) / len(roe_values)

            retention_ratio = 1 - info['payout_ratio']
            growth_rate = retention_ratio * roe
            
            results = {
                'Growth Rate': growth_rate,
                'Retention Ratio': retention_ratio,
                'ROE': roe,
            }            
            return results
            
        except Exception as e:
            print(f"성장률 계산 중 오류 발생: {str(e)}")
            return None
    
# 편의를 위한 함수형 인터페이스
def calculate_growth_rate(ticker_symbol: str, period: str = "annual") -> Dict[str, float]:
    calculator = GrowthCalculatorShareholder(ticker_symbol)
    return calculator.calculate_growth_rate(period)

if __name__ == "__main__":
    # 삼성전자의 성장률 계산
    ticker = "005930.KS"
    results = calculate_growth_rate(ticker) 