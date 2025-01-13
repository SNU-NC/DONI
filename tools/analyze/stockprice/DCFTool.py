from typing import Dict, Tuple
from tools.DCF.calculators.fcfe_calculator import FCFECalculator
from tools.DCF.calculators.wacc_calculator import WACCCalculator
from tools.DCF.calculators.growth_calculator_shareholder import GrowthCalculatorShareholder
from tools.DCF.collectors.info_data_collector import InfoDataCollector
from tools.DCF.collectors.financial_data_collector import FinancialDataCollector
import pandas as pd
from typing import Tuple, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from typing import Optional, Dict, Any







##Tool 설명
_DCF_DESCRIPTION = """
DCF(company_name: str) -> str:
"""


class DCF:
    """회사 가치를 계산하는 클래스"""
    
    def __init__(self, company_name: str):
        """
        company_name을 입력받아 kospi_list.csv에서 종목코드를 찾아 ticker_symbol로 변환.
        
        Args:
            company_name (str): 회사명
            kospi_list_path (str): kospi_list.csv 파일 경로
        """
        kospi_list = pd.read_csv('data/kospi_list.csv')
        #회사명을 기반으로 종목코드 검색
        row = kospi_list[kospi_list['종목명'] == company_name]
        # if row.empty:
        #     raise ValueError(f"Company name '{company_name}' not found in kospi_list.csv.")
        # 종목코드 추출 및 Ticker Symbol 생성
        ticker_code = row['종목코드'].values[0]
        ticker_symbol = f"{ticker_code}.KS"
        
        self.ticker_symbol = ticker_symbol
        self.fcfe_calculator = FCFECalculator(ticker_symbol)
        self.wacc_calculator = WACCCalculator(ticker_symbol)
        self.net_income_growth_calculator = GrowthCalculatorShareholder(ticker_symbol)
        self.financial_data_collector = FinancialDataCollector(ticker_symbol)
        self.info_collector = InfoDataCollector(ticker_symbol)
        self.shares_outstanding = self.info_collector.get_info()['shares_outstanding']
        #print(f"Shares Outstanding: {self.shares_outstanding}")
    
    def calculate_10year_present_value(
            self,
            fcfe: float = None, 
            cost_of_equity: float = None,
            net_income_growth_rate: float = None,
            retention_ratio: float = None
            ) -> float:
        """향후 10년 주주가치의 총현재가치 계산
        
        Args:
            period (str): 기간 (annual, quarterly)
            fcfe (float): FCFE
            cost_of_equity (float): 자본비용
            net_income_growth_rate (float): 순이익 성장률
            retention_ratio (float): 이익 중 재투자율
        Returns:
            float: 향후 10년 주주가치의 총 현재가치
        """

        # metrics = self.financial_data_collector.extract_financial_metrics(period)

        # # 4년 평균 순이익 계산
        # net_income_values = metrics['net_income'].iloc[:min(4, len(metrics))]
        # net_income = net_income_values.mean()

        
        # FCFE가 net income growth만큼 성장한다고 가정
        after_fcfe = []
        for i in range(10):
            after_fcfe.append(fcfe * ((1 + net_income_growth_rate) ** (i+1)))
 
        after_10year_fcfe = after_fcfe[9]

        # print(f"_1year_fcfe: {_1year_fcfe}")
        # print(f"_2year_fcfe: {_2year_fcfe}")
        # print(f"_3year_fcfe: {_3year_fcfe}")
        # print(f"_4year_fcfe: {_4year_fcfe}")
        # print(f"_5year_fcfe: {_5year_fcfe}")

        after_fcfe_present_value = []
        for i in range(10):
            after_fcfe_present_value.append(after_fcfe[i] / ((1 + cost_of_equity) ** (i+1)))

        total_present_value = sum(after_fcfe_present_value)

        # print(f"Total Present Value: {total_present_value}")

        return total_present_value, after_10year_fcfe
        
    def calculate_terminal_value(
            self,
            cost_of_equity: float = None,
            retention_ratio: float = None,
            after_10year_fcfe: float = None
            ) -> Tuple[float, float, float, float, float]:
        """Terminal Value 계산
        
        Args:
            period (str): 기간 (annual, quarterly)
            fcfe (float): FCFE
            cost_of_equity (float): 자본비용
            net_income_growth_rate (float): 순이익 성장률
            retention_ratio (float): 이익 중 재투자율
        Returns:
            Tuple[float, float, float, float, float]: Terminal Value 계산 결과
        """
        growth_rate_tv = 0.0 # Terminal Value 계산에 사용할 성장률 0%로 고정(성숙기업 무성장 예상)

        _11year_fcfe = after_10year_fcfe * (1 + growth_rate_tv) * retention_ratio
        terminal_value = _11year_fcfe / (cost_of_equity - growth_rate_tv)
        terminal_value_pv = terminal_value / ((1 + cost_of_equity) ** 10)
        # print(f"_11year_fcfe: {_11year_fcfe}")
        # print(f"terminal_value: {terminal_value}")
        
        return terminal_value_pv, growth_rate_tv
    
    def calculate_total_value(self, period: str = "annual") -> Dict[str, float]:
        """총가치 계산"""
        calculated_fcfe = self._calculate_fcfe(period)
        fcfe = calculated_fcfe['FCFE']

        calculated_wacc = self._calculate_wacc()
        cost_of_equity = calculated_wacc['Cost of Equity']

        calculated_net_income_growth_rate = self._calculate_net_income_growth_rate()
        net_income_growth_rate = calculated_net_income_growth_rate['Growth Rate']
        retention_ratio = calculated_net_income_growth_rate['Retention Ratio']

        _10year_pv, after_10year_fcfe = self.calculate_10year_present_value(fcfe, cost_of_equity, net_income_growth_rate, retention_ratio)
        terminal_value_pv, growth_rate_tv = self.calculate_terminal_value(cost_of_equity, net_income_growth_rate, retention_ratio, after_10year_fcfe)

        total_value = _10year_pv + terminal_value_pv

        # print(f"Total Value: {total_value}")
        return total_value, fcfe, cost_of_equity, net_income_growth_rate, growth_rate_tv
    
    def calculate_per_share(self, period: str = "annual") -> Dict[str, float]:
        """주당가치 계산"""
        # 실제 주가 가져오기
        actual_price = self.info_collector.get_info().get('regularMarketPreviousClose', 0)
        
        results = {}
        best_result = None
        min_diff = float('inf')
        
        # 1~4년 평균으로 계산
        for years in range(1, 5):
            try:
                calculated_fcfe = self._calculate_fcfe(period, years)
                fcfe = calculated_fcfe['FCFE']
                
                calculated_wacc = self._calculate_wacc()
                cost_of_equity = calculated_wacc['Cost of Equity']
                
                calculated_growth = self._calculate_net_income_growth_rate(period, years)
                growth_rate = calculated_growth['Growth Rate']
                retention_ratio = calculated_growth['Retention Ratio']
                roe = calculated_growth['ROE']

                if fcfe <= 0 or roe <= 0:
                    results[f'result_{years}year'] = None
                    continue
                
                _10year_pv, after_10year_fcfe = self.calculate_10year_present_value(
                    fcfe=fcfe,
                    cost_of_equity=cost_of_equity,
                    net_income_growth_rate=growth_rate,
                    retention_ratio=retention_ratio
                )
                
                terminal_value_pv, growth_rate_tv = self.calculate_terminal_value(
                    cost_of_equity=cost_of_equity,
                    retention_ratio=retention_ratio,
                    after_10year_fcfe=after_10year_fcfe
                )
                
                total_value = _10year_pv + terminal_value_pv
                per_share = total_value / self.shares_outstanding
                
                # 결과 저장
                results[f'result_{years}year'] = {
                    'per_share': per_share,
                    'cost_of_equity': cost_of_equity,
                    'growth_rate': growth_rate,
                    'ratio_capex_ocf': calculated_fcfe['Ratio CapEx/OCF'],
                    'ratio_repayment_issuance': calculated_fcfe['Ratio Repayment/Issuance']
                }
                # print("="*100)
                # print(f"{years}년 평균 주당 가치 계산 결과: {results[f'result_{years}year']['per_share']}\n")
                
                # 실제 주가와의 차이 계산
                diff = abs(per_share - actual_price)
                if diff < min_diff:
                    min_diff = diff
                    best_result = results[f'result_{years}year']
                
            except Exception as e:
                # print(f"{years}년 평균 계산 중 오류 발생: {str(e)}")
                continue

        if best_result is None:
            # print("="*100)
            # print("기업의 최근 4년간 ROE, FCFE가 0이거나 음수인 경우, 기업의 영속성을 담보할 수 없기 때문에, DCF 계산이 불가능합니다.")
            # print("="*100)
            return None
        
        # print("="*100)
        # print(f"actual_price: {actual_price}")
        # print(f"Best Result: {best_result['per_share']}")
        # # print("="*100)
        # for key, value in best_result.items():
        #     print(f"{key}: {value}")
        best_result=best_result['per_share']
            
        return best_result
    
    def _calculate_fcfe(self, period: str = "annual", years: int = None) -> Dict[str, float]:
        """FCFE 계산"""
        return self.fcfe_calculator.calculate_fcfe(period, years)
    
    def _calculate_wacc(self) -> float:
        """WACC 계산"""
        return self.wacc_calculator.calculate_wacc()
    
    def _calculate_net_income_growth_rate(self, period: str = "annual", years: int = None) -> Dict[str, float]:
        """성장률 계산"""
        return self.net_income_growth_calculator.calculate_net_income_growth_rate(period, years)
    
class DCFInputs(BaseModel):
    company_name: str = Field(..., description="검색하고자 하는 기업명")
   


class DCFTool(BaseTool):
    name: str = "DCF"
    description: str = _DCF_DESCRIPTION
    args_schema: Type[BaseModel] = DCFInputs

    def _run(self, company_name: str) -> str:
        try:


            # ValuationCalculator 초기화
            calculator = DCF(company_name)

            # 주당 가치 계산
            best_result = calculator.calculate_per_share()

            if best_result is None:
                return f"기업 {company_name}의 DCF 기반 가치 평가를 수행할 수 없습니다. ROE 또는 FCFE 값이 유효하지 않습니다."

            # 결과 메시지 생성
            result = (
                f"DCF(Discounted Cash Flow) 기반 가치평가 방식은 다음과 같습니다:\n"
                f"1. FCFE(자유현금흐름)를 추정하여 향후 10년간의 현금 흐름을 예측합니다.\n"
                f"2. 각 연도의 FCFE를 자본비용(WACC)으로 할인하여 총 현재가치를 계산합니다.\n"
                f"3. 마지막으로 Terminal Value를 계산하고 이를 합산하여 기업의 총 가치를 도출합니다.\n"
                f"4. 기업의 총 가치를 주식 수로 나누어 주당 가치를 계산합니다.\n\n"
                f"매칭된 기업명: {company_name}\n"
                f"최종 DCF 기반 주당 가치(Best Result): {best_result:.2f}원"
            )

            return result

        except Exception as e:
            return f"예상치 못한 오류가 발생했습니다: {str(e)}"