from typing import Dict, Optional
from tools.DCF.collectors.financial_data_collector import FinancialDataCollector

class FCFECalculator:
    """FCFE(Free Cash Flow to Equity)를 계산하는 클래스"""
    
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.financial_collector = FinancialDataCollector(ticker_symbol)

    def calculate_fcfe(self, period: str = "annual", years: int = 1) -> Dict[str, float]:
        """FCFE를 계산하는 메서드
        Args:
            period (str): 기간 (annual, quarterly)
            years (int): 평균을 계산할 연도 수
        Returns:
            Dict[str, float]: FCFE 계산 결과
            
            FCFE (float): Free Cash Flow to Equity 
                : 주주들이 실질적으로 가질 수 있는 현금흐름.
                : 회사가 번 돈에서 필요한 비용을 빼고 남은 돈(주주에게 돌아갈 수 있는 돈)

            Operating Cash Flow (float): 영업활동현금흐름
                : 회사가 영업을 통해 얼마나 많은 현금을 벌었는지 나타냄
            Capital Expenditure (float): 자본지출
                : 회사가 자산을 구매하거나 유지하기 위해 사용한 현금
            Repayment of Debt (float): 부채상환(-)
                : 부채를 상환하면 현금이 줄어듦
            Issuance of Debt (float): 부채발행(+)
                : 부채를 발행하면 현금이 증가함
        """
        try:
            # 재무 지표 수집
            metrics = self.financial_collector.extract_financial_metrics(period)
            
            # n년 평균 FCFE 계산
            fcfe_values = []
            for i in range(min(years, len(metrics['operating_cash_flow']))):
                fcfe = (metrics['operating_cash_flow'].iloc[i] 
                        - metrics['capital_expenditure'].iloc[i] 
                        + metrics['repayment_of_debt'].iloc[i]
                        + metrics['issuance_of_debt'].iloc[i])
                fcfe_values.append(fcfe)
            
            fcfe_average = sum(fcfe_values) / len(fcfe_values)
            
            ratio_capex_ocf_values = []
            ratio_repayment_issuance_values = []
            for i in range(min(years, len(metrics['capital_expenditure']))):
                ratio_capex_ocf = metrics['capital_expenditure'].iloc[i] / metrics['operating_cash_flow'].iloc[i]
                ratio_repayment_issuance = -metrics['repayment_of_debt'].iloc[i] / metrics['issuance_of_debt'].iloc[i]
                ratio_capex_ocf_values.append(ratio_capex_ocf)
                ratio_repayment_issuance_values.append(ratio_repayment_issuance)
            
            ratio_capex_ocf = sum(ratio_capex_ocf_values) / len(ratio_capex_ocf_values)
            ratio_repayment_issuance = sum(ratio_repayment_issuance_values) / len(ratio_repayment_issuance_values)

            # print(f"Capital Expenditure / Operating Cash Flow: {ratio_capex_ocf:.2%}")
            # print(f"부채상환 / 부채발행: {ratio_repayment_issuance:.2%}")
            
            results = {
                'FCFE': float(fcfe_average),
                'Operating Cash Flow': float(metrics['operating_cash_flow'].iloc[0]),
                'Capital Expenditure': float(metrics['capital_expenditure'].iloc[0]),
                'Repayment of Debt': float(metrics['repayment_of_debt'].iloc[0]),
                'Issuance of Debt': float(metrics['issuance_of_debt'].iloc[0]),
                'Ratio CapEx/OCF': float(ratio_capex_ocf),
                'Ratio Repayment/Issuance': float(ratio_repayment_issuance)
            }
            
            return results
            
        except Exception as e:
            print(f"FCFE 계산 중 오류 발생: {str(e)}")
            return None

# 편의를 위한 함수형 인터페이스
def calculate_fcfe(ticker_symbol: str, period: str = "annual", years: int = 1) -> Dict[str, float]:
    calculator = FCFECalculator(ticker_symbol)
    return calculator.calculate_fcfe(period, years)

if __name__ == "__main__":
    # 삼성전자의 FCFF 계산
    ticker = "005930.KS"
    results = calculate_fcfe(ticker)
    
    if results:
        print(f"\n{ticker}의 FCFE 계산 결과:")
        for key, value in results.items():
            print(f"{key}: {value:,.0f}") 