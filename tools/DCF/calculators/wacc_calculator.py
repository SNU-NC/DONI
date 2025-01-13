from typing import Dict, Optional
from tools.DCF.collectors.financial_data_collector import FinancialDataCollector
from tools.DCF.collectors.market_data_collector import MarketDataCollector
from tools.DCF.utils.financial_utils import get_yfinance_beta, calculate_beta
import pandas as pd

class WACCCalculator:
    """WACC(Weighted Average Cost of Capital)를 계산하는 클래스"""
    
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.financial_collector = FinancialDataCollector(ticker_symbol)
        self.market_collector = MarketDataCollector()
        self.effective_tax_rate = None
    
    def calculate_wacc(self) -> Dict[str, float]:
        """WACC 계산"""
        try:
            # 재무 지표 수집
            metrics = self.financial_collector.extract_financial_metrics()
            
            # Series 객체에서 값 추출
            metrics = {key: value.iloc[0] if isinstance(value, pd.Series) else value for key, value in metrics.items()}
            
            # 실효세율 먼저 계산
            self.effective_tax_rate = self._calculate_effective_tax_rate(metrics)
            
            # 자본비용 계산
            cost_of_equity = self._calculate_cost_of_equity()
            
            # 부채비용 계산
            cost_of_debt = self._calculate_cost_of_debt(metrics)
            
            # 자본구조 계산
            equity_weight, debt_weight = self._calculate_capital_structure(metrics)
            
            # WACC 계산
            wacc = (cost_of_equity * equity_weight + 
                   cost_of_debt * (1 - self.effective_tax_rate) * debt_weight)
            
            results =  {
                'WACC': wacc,
                'Cost of Equity': cost_of_equity,
                'Cost of Debt': cost_of_debt,
                'Equity Weight': equity_weight,
                'Debt Weight': debt_weight,
                'Risk-free Rate': self.risk_free_rate,
                'Market Risk Premium': self.market_risk_premium,
                'Beta': self.beta,
                'Effective Tax Rate': self.effective_tax_rate
            }
            
            # for key, value in results.items():
            #     if key == 'Beta':
            #         print(f"{key}: {value:.2f}")
            #     else:
            #         print(f"{key}: {value:.2%}")
            
            return results
            
        except Exception as e:
            print(f"WACC 계산 중 오류 발생: {str(e)}")
            return None
    
    def _calculate_cost_of_equity(self) -> float:
        """자본비용 계산 (CAPM 모델 사용)"""
        # 무위험수익률 조회
        self.risk_free_rate = self.market_collector.get_risk_free_rate()
        # print(f"무위험수익률: {self.risk_free_rate:.2%}")
        
        # 시장위험프리미엄 계산
        self.market_risk_premium = self.market_collector.get_market_risk_premium()
        # print(f"시장위험프리미엄: {self.market_risk_premium:.2%}")
        
        # 베타 계산
        self.beta = self._get_beta()
        # print(f"베타: {self.beta:.2f}")
        
        # CAPM 모델로 자본비용 계산
        return self.risk_free_rate + (self.beta * self.market_risk_premium)
    
    def _calculate_cost_of_debt(self, metrics: Dict[str, float]) -> float:
        """부채비용 계산 (세후 기준)"""
        try:
            interest_expense = abs(metrics['interest_expense'])
            total_debt = metrics['total_debt']
            
            if total_debt == 0:
                return 0
            
            # 세후 부채비용 계산
            return interest_expense / total_debt
            
        except Exception:
            return 0.045  # 기본값
    
    def _calculate_capital_structure(self, metrics: Dict[str, float]) -> tuple[float, float]:
        """자본구조 계산 (장부가 기준)"""
        try:
            total_equity = metrics['total_equity']
            total_debt = metrics['total_liabilities_net_minority_interest']
            
            total_value = total_equity + total_debt
            
            equity_weight = total_equity / total_value
            debt_weight = total_debt / total_value
            
            return equity_weight, debt_weight
            
        except Exception as e:
            print(f"자본구조 계산 중 오류 발생: {str(e)}")
            return 0.5, 0.5
    
    def _calculate_effective_tax_rate(self, metrics: Dict[str, float]) -> float:
        """실효세율 계산"""
        try:
            pretax_income = metrics['pretax_income']
            income_tax = metrics['tax_provision']
            
            # if pretax_income == 0:
            #     return 0.22
            
            effective_tax_rate = income_tax / pretax_income
            print(f"실효세율: {effective_tax_rate:.2%}")

            if not (0 <= effective_tax_rate <= 1):
                return 0.22
                
            return effective_tax_rate
            
        except Exception:
            return 0.22
    
    def _get_beta(self) -> float:
        """베타값 조회 또는 계산"""
        # yfinance에서 베타값 조회 시도
        beta = get_yfinance_beta(self.ticker_symbol)
        if beta is not None:
            return beta
        
        # 직접 계산
        return calculate_beta(self.ticker_symbol) 