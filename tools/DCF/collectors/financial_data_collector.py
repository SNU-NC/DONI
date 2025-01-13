import yfinance as yf
import pandas as pd
from typing import Dict, Optional, Tuple

pd.set_option('future.no_silent_downcasting', True)

class FinancialDataCollector:
    """재무제표 데이터를 수집하는 클래스"""
    
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.stock = yf.Ticker(ticker_symbol)
    
    def get_financial_statements(self, period: str = "annual") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """재무제표 데이터를 가져오는 메서드
        
        args:
            period (str): 주기 (annual, quarterly)

        returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 재무제표 데이터
        """
        if period == "annual":
            income_stmt = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            cash_flow = self.stock.cashflow
        else:
            income_stmt = self.stock.quarterly_financials
            balance_sheet = self.stock.quarterly_balance_sheet
            cash_flow = self.stock.quarterly_cashflow
            
        return income_stmt, balance_sheet, cash_flow
    
    def extract_financial_metrics(self, period: str = "annual") -> Dict[str, pd.Series]:
        """필요한 재무 지표들을 추출하는 메서드"""
        income_stmt, balance_sheet, cash_flow = self.get_financial_statements(period)
        metrics = {}

        # EBIT 추출
        ebit_keys = ['EBIT']
        metrics['ebit'] = self._find_metric(income_stmt, ebit_keys, "EBIT")

        # EBITDA 추출
        try:
            metrics['ebitda'] = income_stmt.loc['EBITDA']
        except KeyError:
            raise KeyError("EBITDA 데이터를 찾을 수 없습니다.")
        
        # 법인세 비용 추출
        tax_keys = ['Tax Provision']
        metrics['tax_provision'] = self._find_metric(income_stmt, tax_keys, "법인세 비용")
        
        # 당기순이익 추출
        try:
            metrics['net_income'] = income_stmt.loc['Net Income']
        except KeyError:
            raise KeyError("Net Income 데이터를 찾을 수 없습니다.")
        
        # 총자산 추출
        try:
            metrics['total_assets'] = balance_sheet.loc['Total Assets']
        except KeyError:
            raise KeyError("Total Assets 데이터를 찾을 수 없습니다.")
        
        # 재무상태표 지표 추출
        metrics.update(self._get_balance_sheet_metrics(balance_sheet))
        
        # 현금흐름표 지표 추출
        metrics.update(self._get_cash_flow_metrics(cash_flow))
        
        return metrics
    
    def _get_balance_sheet_metrics(self, balance_sheet: pd.DataFrame) -> Dict[str, pd.Series]:
        """재무상태표 관련 지표 추출"""
        metrics = {}
        
        # 자본총계 추출
        equity_keys = ['Total Equity Gross Minority Interest']
        metrics['total_equity'] = self._find_metric(balance_sheet, equity_keys, "자본총계")
        
        # 부채총계 추출(Total Liabilities Net Minority Interest)
        total_liabilities_net_minority_interest_keys = ['Total Liabilities Net Minority Interest']
        metrics['total_liabilities_net_minority_interest'] = self._find_metric(balance_sheet, total_liabilities_net_minority_interest_keys, "부채총계")

        # 부채총계 추출(Total Debt)
        total_debt_keys = ['Total Debt']
        try:
            metrics['total_debt'] = self._find_metric(balance_sheet, total_debt_keys, "부채총계")
        except KeyError:
            # 유동부채와 비유동부채 합산
            current = self._find_metric(balance_sheet, ['Total Current Liabilities'], "유동부채")
            non_current = self._find_metric(balance_sheet, ['Total Non Current Liabilities'], "비유동부채")
            metrics['total_debt'] = current + non_current
        
        # 유동자산 추출
        current_assets_keys = ['Total Current Assets', 'Current Assets', 'Total Current Assets Gross']
        metrics['current_assets'] = self._find_metric(balance_sheet, current_assets_keys, "유동자산")
        
        # 유동부채 추출
        current_liabilities_keys = ['Total Current Liabilities', 'Current Liabilities', 'Total Current Liabilities Net']
        metrics['current_liabilities'] = self._find_metric(balance_sheet, current_liabilities_keys, "유동부채")
        
        # 현금 및 현금성자산 추출
        cash_keys = ['Cash And Cash Equivalents', 'Cash And Short Term Investments', 'Cash & Equivalents']
        metrics['cash_and_equivalents'] = self._find_metric(balance_sheet, cash_keys, "현금 및 현금성자산")

        # Invested Capital 추출
        invested_capital_keys = ['Invested Capital']
        metrics['invested_capital'] = self._find_metric(balance_sheet, invested_capital_keys, "투하자본")
        
        return metrics
    
    def _get_cash_flow_metrics(self, cash_flow: pd.DataFrame) -> Dict[str, pd.Series]:
        """현금흐름표 관련 지표 추출"""
        metrics = {}
        
        # 자본적 지출 추출
        capex_keys = ['Capital Expenditure', 'Capital Expenditures', 'CapEx']
        metrics['capital_expenditure'] = self._find_metric(cash_flow, capex_keys, "자본적 지출").abs()
        
        # operating cash flow 추출
        operating_cash_flow_keys = ['Operating Cash Flow']
        metrics['operating_cash_flow'] = self._find_metric(cash_flow, operating_cash_flow_keys, "영업활동현금흐름")

        # Repayment of Debt 추출
        repayment_of_debt_keys = ['Repayment Of Debt']
        metrics['repayment_of_debt'] = self._find_metric(cash_flow, repayment_of_debt_keys, "부채상환")
        if metrics['repayment_of_debt'].isna().any():
            metrics['repayment_of_debt'] = pd.to_numeric(metrics['repayment_of_debt'].fillna(0))
        
        # Issuance of Debt 추출
        issuance_of_debt_keys = ['Issuance Of Debt']
        metrics['issuance_of_debt'] = self._find_metric(cash_flow, issuance_of_debt_keys, "부채발행")
        if metrics['issuance_of_debt'].isna().any():
            metrics['issuance_of_debt'] = pd.to_numeric(metrics['issuance_of_debt'].fillna(0))

        return metrics
    
    def _find_metric(self, statement: pd.DataFrame, keys: list, metric_name: str) -> pd.Series:
        """주어진 키 목록에서 재무 지표를 찾는 헬퍼 메서드"""
        for key in keys:
            try:
                return statement.loc[key]
            except KeyError:
                continue
        raise KeyError(f"{metric_name} 데이터를 찾을 수 없습니다.") 