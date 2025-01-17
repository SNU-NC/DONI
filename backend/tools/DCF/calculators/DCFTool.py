from typing import Dict, Tuple
from tools.DCF.calculators.fcfe_calculator import FCFECalculator
from tools.DCF.calculators.wacc_calculator import WACCCalculator
from tools.DCF.calculators.growth_calculator_shareholder import GrowthCalculatorShareholder
from tools.DCF.collectors.info_data_collector import InfoDataCollector
from tools.DCF.collectors.financial_data_collector import FinancialDataCollector
import pandas as pd
from typing import Tuple, Type
from langchain_core.tools import BaseTool
import json
import unicodedata
from rapidfuzz import process, fuzz
from langchain.pydantic_v1 import BaseModel, Field

def normalize_str(s):
    # 유니코드 정규화 (NFC)
    return unicodedata.normalize('NFC', s)

def load_company_names(file_path="/Users/jangeunji/Downloads/all_company_names.json"):    ####all_company_names에 있는 json file 경로.
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    companies = data["companies"]
    # 모든 기업명을 NFC 정규화
    companies = [normalize_str(c) for c in companies]
    return companies

# 한글 음차 -> 영어 알파벳 매핑
kor_to_eng = {
    "에이": "a", "비": "b", "씨": "c", "디": "d", "이": "e", "에프": "f",
    "지": "g", "에이치": "h", "아이": "i", "제이": "j", "케이": "k",
    "엘": "l", "엠": "m", "엔": "n", "오": "o", "피": "p", "큐": "q",
    "아르": "r", "에스": "s", "티": "t", "유": "u", "브이": "v",
    "더블유": "w", "엑스": "x", "와이": "y", "제트": "z", "네이버": "NAVER"
}

eng_to_kor = {v: k for k, v in kor_to_eng.items()}

def apply_mapping(query, mapping):
    items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
    result = query
    for k_str, v_str in items:
        if k_str in result:
            result = result.replace(k_str, v_str)
    return result

def generate_candidates(query):
    candidates = [query]
    # 한글 -> 영문 변환
    kor_to_eng_query = apply_mapping(query, kor_to_eng)
    if kor_to_eng_query != query:
        candidates.append(kor_to_eng_query)

    # 영문 -> 한글 변환
    eng_to_kor_query = query
    for eng_char, kor_str in eng_to_kor.items():
        if eng_char in eng_to_kor_query:
            eng_to_kor_query = eng_to_kor_query.replace(eng_char, kor_str)
    if eng_to_kor_query != query:
        candidates.append(eng_to_kor_query)

    # 중복 제거
    candidates = list(set(candidates))
    return candidates

def match_company_name(query, companies, limit=1):
    query_norm = normalize_str(query)
    candidates = generate_candidates(query_norm)
    
    all_results = []
    for candidate in candidates:
        candidate_norm = normalize_str(candidate)
        results = process.extract(candidate_norm, companies, scorer=fuzz.WRatio, limit=limit)
        all_results.extend(results)
    
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    
    seen = set()
    unique_results = []
    for match, score, idx in all_results:
        if match not in seen:
            unique_results.append((match, score, idx))
            seen.add(match)
        if len(unique_results) == limit:
            break
    
    return unique_results[0] if unique_results else None







##Tool 설명
_DCF_DESCRIPTION = """
DCF(company_name: str) -> str:
- company_name**: 가치 평가를 수행할 기업의 이름입니다.
- 기업의 FCFE, WACC, 순이익 성장률 등 재무 데이터를 활용하여 10년 미래 현금흐름과 터미널 가치를 계산하고, 이를 통해 기업의 총 가치와 주당 가치를 산출합니다. 
- 결과를 문자열로 반환합니다.
- 기업의 내재 가치 평가나 DCF 모델을 사용한 주당 가치 분석이 필요할 때 이 Tool을 사용합니다.
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
        kospi_list = pd.read_csv('kospi_list.csv')
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
    company_name: str = Field(..., description="분석할 회사명")


class DCFTool(BaseTool):
    name: str = "DCF"
    description: str = _DCF_DESCRIPTION
    args_schema: Type[BaseModel] = DCFInputs

    def _run(self, company_name: str) -> str:
        try:
            # 기업명 유사도 검색
            companies = load_company_names()
            best_match = match_company_name(company_name, companies)

            if not best_match:
                return f"'{company_name}'와 유사한 기업명을 찾을 수 없습니다. 다시 확인해 주세요."

            # 유사도가 가장 높은 기업명을 사용
            matched_name = best_match[0]
            print(f"입력된 기업명: {company_name}, 매칭된 기업명: {matched_name}")

            # ValuationCalculator 초기화
            calculator = DCF(matched_name)

            # 주당 가치 계산
            best_result = calculator.calculate_per_share()

            if best_result is None:
                return f"기업 {matched_name}의 DCF 기반 가치 평가를 수행할 수 없습니다. ROE 또는 FCFE 값이 유효하지 않습니다."

            # 결과 메시지 생성
            result = (
                f"DCF(Discounted Cash Flow) 기반 가치평가 방식은 다음과 같습니다:\n"
                f"1. FCFE(자유현금흐름)를 추정하여 향후 10년간의 현금 흐름을 예측합니다.\n"
                f"2. 각 연도의 FCFE를 자본비용(WACC)으로 할인하여 총 현재가치를 계산합니다.\n"
                f"3. 마지막으로 Terminal Value를 계산하고 이를 합산하여 기업의 총 가치를 도출합니다.\n"
                f"4. 기업의 총 가치를 주식 수로 나누어 주당 가치를 계산합니다.\n\n"
                f"입력된 기업명: {company_name}\n"
                f"매칭된 기업명: {matched_name}\n"
                f"최종 DCF 기반 주당 가치(Best Result): {best_result:.2f}원"
            )

            return result

        except Exception as e:
            return f"예상치 못한 오류가 발생했습니다: {str(e)}"