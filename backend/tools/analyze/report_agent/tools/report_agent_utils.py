from yahooquery import search
from deep_translator import GoogleTranslator
from typing import List
import pandas as pd
from datetime import datetime

# 한글 문자 범위를 이용해 한글 포함 여부 확인
def _contains_korean(text):
    for char in text:
        if '\uac00' <= char <= '\ud7a3' or '\u3131' <= char <= '\u318e':
            return True
    return False

def get_ticker(company_name):
    try:            
        # 한글 포함 여부 확인
        is_korean = _contains_korean(company_name)

        if is_korean:
            # 회사명을 영어로 번역
            translated = GoogleTranslator(source='auto', target='en').translate(company_name)
            
            # 번역된 이름으로 검색
            results = search(translated)
        else:
            results = search(company_name)
            
        # 거래소 우선순위 정의
        priority_exchanges = ['KSC', 'NYQ', 'NMS', 'JPX', 'HKG']
        
        # 각 거래소별로 순차 검색
        for exchange in priority_exchanges:
            for quote in results['quotes']:
                if quote['exchange'] == exchange:
                    return quote['symbol']
        
        # 해당하는 거래소가 없는 경우        
        return None
    except Exception as e:
        print(f"Error translating or searching for {company_name}: {e}")
        return None
    
def extract_segment(income_stmt_cum: pd.DataFrame) -> List[str]:
    segments = []
    for item in income_stmt_cum.iloc[:, 0]:
        if item == '영업수익':
            break
        segments.append(item)
    return segments

def yoy_calculator(income_stmt_cum: pd.DataFrame) -> pd.Series:
    present_quarter = income_stmt_cum.iloc[:,1+4]
    year_ago_quarter = income_stmt_cum.iloc[:,1]
    yoy = ((present_quarter - year_ago_quarter) / year_ago_quarter) * 100
    return yoy

class consensusCalculator:
    def __init__(self, income_stmt_cum: pd.DataFrame, current_quarter_sales_consensus:float, state):
        self.income_stmt_cum = income_stmt_cum
        self.current_quarter_sales_consensus = current_quarter_sales_consensus
        self.state = state

    def _current_quarter_sales_cum_consensus_calculator(self) -> float:
        """네이버 증권 -> 종목분석 -> 컨센서스 -> 어닝서프라이즈(매출액) -> 분기
            return :
            직전 분기 매출액에 대한 발표직전 컨센서스 (단위: 억원) + 직전 분기의 전분기까지의 실제 매출액 = 직전 분기까지 누적 매출액 컨센서스
        """
        
        before_current_quarter_sales_cum = self.income_stmt_cum.loc[self.income_stmt_cum['계정'] == '영업수익'].iloc[:,4].values[0]
        current_quarter_sales_cum_consensus = self.current_quarter_sales_consensus + before_current_quarter_sales_cum

        return current_quarter_sales_cum_consensus
    
    def calculate(self) -> float:
        current_quarter_sales_cum_consensus = self._current_quarter_sales_cum_consensus_calculator()

        year_ago_quarter_sales_cum = self.income_stmt_cum.loc[self.income_stmt_cum['계정'] == '영업수익'].iloc[:,1].values[0]
        yoy_consensus = ((current_quarter_sales_cum_consensus - year_ago_quarter_sales_cum) / year_ago_quarter_sales_cum) * 100

        return yoy_consensus, current_quarter_sales_cum_consensus


def combine_report(state):
    """리포트 최종 출력"""
    title = f"{state.company_name} 리포트"
    report = ""
    report += f"{title}\n"
    report += f"="*len(title)*2
    report += f"\n\n"
    for key, value in state.result.items():
        report += f"{value}\n\n"
        report += f"-"*50
        report += f"\n\n"

    report += f"\n"
    report += "본 자료는 사업보고서, 기사 등 신뢰할 수 있는 자료를 바탕으로 작성되었습니다.\n"
    report += "그러나, AI의 의견이 반영되었으며 정확성이나 완전성을 보장할 수 없습니다.\n"
    report += "본 자료는 참고용으로 투자자 자신의 판단과 책임하에 투자하시기 바랍니다.\n"
    report += f"최종 리포트 작성일 : {datetime.now().strftime('%Y-%m-%d')}"
    return report