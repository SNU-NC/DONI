"""
마켓 데이터 분석을 위한 도구 모듈
"""
import os
from typing import Type, Optional, Dict, Any, List, Union
from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from tools.marketData.prompts import EXCHANGE_RATE_TOOL_PREFIX, STOCK_PRICE_TOOL_PREFIX
import OpenDartReader

dart = OpenDartReader("4925a6e6e69d8f9138f4d9814f56f371b2b2079a")

class StockPriceInputSchema(BaseModel):
    query: str = Field(..., description="주가 데이터에서 조회할 정보 (예: 최근 1년간 주가 추이, 최근 3개월간 주가 변동률, 00년 00월 00일 종가...)")
    symbols: str = Field(..., description="주가 데이터를 조회할 주식종목코드 또는 기업명 (예: '005930' 또는 '삼성전자'...)")
    start_date: str = Field(default="", description="조회 시작일 (YYYY-MM-DD 형식, 빈 문자열이면 1년전...)")
    end_date: str = Field(default="", description="조회 종료일 (YYYY-MM-DD 형식, 빈 문자열이면 오늘...)")

class ExchangeRateInputSchema(BaseModel):
    query: str = Field(..., description="환율 데이터와 관련된 쿼리(예: 00년 00월 00일 환율 정보, 과거 0년/0개월간 환율 변화 추세)")

class GetStockCodeSchema(BaseModel):
    company: str = Field(..., description="정확한 기업 이름(예: 삼성전자, 교촌에프엔비, 유니온, 깨끗한나라, POSCO홀딩스)")

class StockPriceTool(BaseTool):
    """주가 데이터 조회 도구"""
    name: str = "stock_price_tool"
    description: str = "개별 종목의 주가 데이터를 조회합니다."
    args_schema: Type[BaseModel] = StockPriceInputSchema
    return_direct: bool = True
    llm: Any = Field(default=None, exclude=True)
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
    def _run(
        self,
        query: str,
        symbols: str,
        start_date: str = "",
        end_date: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            # 한글 문자가 포함되어 있는지 확인
            if any(ord('가') <= ord(c) <= ord('힣') for c in symbols):
                stock_code_tool = GetStockCodeTool()
                symbol = stock_code_tool.invoke({"company":symbols})
            else:
                symbol = symbols

            df = fdr.DataReader(f'KRX:{symbol}', start_date, end_date) # 한국 거래소 데이터 기준
            df = df.rename(columns={
                'Open': '시가',
                'High': '고가', 
                'Low': '저가',
                'Close': '종가',
                'Volume': '거래량',
                'Change': '변동률',
                'UpDown': '등락',
                'Comp': '기업명',
                'Amount': '거래대금',
                'MarCap': '시가총액',
                'Shares': '상장주식수'
            })
            if df.empty:
                return f"• {symbol}: 데이터를 찾을 수 없습니다.\n"
            
            agent = create_pandas_dataframe_agent(
                self.llm,
                prefix=STOCK_PRICE_TOOL_PREFIX,
                df=df,
                verbose=True,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS
            )
            
            query_with_period = f"{query}\n\n참고: 데이터 기간은 {start_date}부터 {end_date}까지입니다."
            result = agent.invoke({"input": query_with_period})
            return result
            
        except Exception as e:
            raise ToolException(f"주가 데이터 조회 중 오류 발생: {str(e)}\n유효한 종목 코드인지 확인해주세요.")
    
class ExchangeRateTool(BaseTool):
    """환율 데이터 조회 도구"""
    name: str = "exchange_rate_tool"
    description: str = "달러/원화 환율 데이터와 관련된 작업을 수행합니다."
    args_schema: Type[BaseModel] = ExchangeRateInputSchema
    return_direct: bool = True

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            df = fdr.DataReader('USD/KRW', start=(datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d'))['Adj Close'].ffill().to_frame()
            llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
            agent = create_pandas_dataframe_agent(
                llm,
                prefix=EXCHANGE_RATE_TOOL_PREFIX,
                df = df,
                verbose=True,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,

            )
            result = agent.invoke({"input": query})
            
            return result
        except Exception as e:
            raise ToolException(f"환율 데이터 조회 중 오류 발생: {str(e)}\n유효한 종목 코드인지 확인해주세요.")

class GetStockCodeTool(BaseTool):
    """종목코드 조회 도구"""
    name: str = "get_stock_code_tool"
    description: str = "정확한 기업 이름을 입력하면 주식코드를 반환합니다."
    args_schema: Type[BaseModel] = GetStockCodeSchema
    return_direct: bool = True

    def _run(
        self,
        company: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        df = dart.corp_codes
        return df[(df['corp_name'] == company) & 
                 (df['stock_code'].notna()) & 
                 (df['modify_date'] == df[df['corp_name'] == company]['modify_date'].max())]['stock_code'].iloc[0]


def create_market_tools() -> List[BaseTool]:
    """마켓 데이터 분석 도구 모음을 생성합니다."""
    return [
        StockPriceTool(),
        ExchangeRateTool(),
    ]