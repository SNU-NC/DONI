from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import pandas as pd
from typing import Optional, Dict, Any
from typing import Type
from pydantic import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
import ast
import yfinance as yf
from config.prompts import _SameSectorComplare_ANL_DESCRIPTION

api_key=os.getenv("OPENAI_API_KEY")
# API 키와 Gateway API 키를 넣습니다.
os.environ["OPENAI_API_KEY"] = api_key


#사전에 만들어진 코스피 기업명과 코드 데이터프레임 
df=pd.read_csv('data/kospi_list.csv')
existing_stock_names = df['종목명'].tolist()


class SameSectorAnalyzer:
    def __init__(self, company_name):
        self.company_name = company_name
        self.company_data = []  # 기업 데이터 저장할 리스트
        self.industry_name = None
        self.total_companies = 0
        self.company_rank = None
        self.driver = None
        self.valid_companies = []
        self.competitors = []
    
    def load_competitor(self):
        try:
            df = pd.read_csv('data/경쟁사데이터.csv')
            if self.company_name in df['종목이름'].values:
                competitors_str = df[df['종목이름'] == self.company_name]['경쟁사'].iloc[0]
                # 문자열을 리스트로 변환
                self.competitors = ast.literal_eval(competitors_str)
            else:
                raise ValueError(f"{self.company_name}에 대한 경쟁사 데이터를 찾을 수 없습니다.")
        except Exception as e:
            #print(f"CSV 파일 로드 중 오류 발생: {e}")
            self.competitors = []
    
    def get_financial_data(self, ticker):
        # 주식 티커에 해당하는 종목 정보 가져오기
        stock = yf.Ticker(ticker)

        # 재무 정보 가져오기
        financials = stock.financials
        info = stock.info

        # 영업이익률 (Operating Margin)
        operating_margin = info.get('operatingMargins', '데이터 없음')

        # 매출액 (Revenue)
        revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else '데이터 없음'

        # 영업이익 (Operating Income)
        operating_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else '데이터 없음'
        # 시가총액 (Market Cap)
        market_cap = info.get('marketCap', '데이터 없음')
        # 매출 증가율 (Revenue Growth)
        revenue_growth = info.get('revenueGrowth', '데이터 없음')

        result = {
            "영업이익률": operating_margin,
            "매출액": revenue,
            "영엽이익": operating_income,
            "매출액증가율": revenue_growth,
            "시가총액": market_cap
        }

        return result

    def analyze_competitors(self):
        # competitors 리스트에 있는 각 기업에 대해 재무 정보 가져오기
        try:
            df2 = pd.read_csv('data/kospi_list.csv')

            for competitor in self.competitors:
                # 종목명으로 종목코드를 찾아서 가져오기
                competitor_code = df2[df2['종목명'] == competitor]['종목코드'].values
                if len(competitor_code) > 0:
                    ticker = competitor_code[0] + '.KS'  # 종목코드
                    
                
                    financial_data = self.get_financial_data(ticker)
                    
                    # 기업명과 함께 재무 데이터를 저장
                    result = {
                        "기업명": competitor,
                        "영업이익률": financial_data["영업이익률"],
                        "매출액": financial_data["매출액"],
                        "영엽이익": financial_data["영엽이익"],
                        "매출액증가율": financial_data["매출액증가율"],
                        "시가총액": financial_data["시가총액"]
                    }

                    # 결과 리스트에 저장
                    self.company_data.append(result)

                    # 결과 출력
                    #print(f"✅ {competitor}의 재무 데이터: {result}")
                else:
                    return e
        except Exception as e:
            print(f"경쟁사 분석 중 오류 발생: {e}")
        
    def get_analyzed_data(self):
        return self.company_data

    def analyze_sector(self):
        if not api_key:
            raise ValueError("OpenAI API 키를 찾을 수 없습니다.")
        llm = ChatOpenAI(
            model_name="gpt-4",
            openai_api_key=api_key
        )

        # 기업 데이터를 잘 포맷팅하여 question에 전달
        formatted_data = "\n".join([", ".join([str(value) for value in row.values()]) for row in self.company_data])
        
        question = f"""
        아래는 경쟁사의 기업 데이터입니다. 기업데이터를 기반으로 경쟁사가 어딘지 알려주세요. 
        경쟁사 대비 매출 비교를 반드시 넣어주세요
        계산하는 부분은 넣지 마시오. 그리고 {self.company_name}의 동향을 비교 분석해 주세요.
        또한 밑에 제시한 예시와 같은 형식으로 작성하시오. 또한, 되도록이면 수치그대로 쓰지말고 단위를 붙여서 보기 쉽도록 하시오

        기업 데이터:
        {formatted_data}
        예시:
        *00기업의 매출액은 경쟁사인 OO, OO, OO와 비교했을 때 00% 높습니다.
        *00기업의 동향 분석: 
          1)00기업의 영업이익은 00원으로 경쟁사 평균인 00원과 비교했을 때 ~한 수치입니다.
          2) 00기업의 영업이익률과 매출액증가율은 00%로 경쟁사인 00%,00%와 비교했을 때 ~입니다.
        *다른 기업과의 비교 분석: 
          1) 업계 전체의 평균적 특징
          2) 약점과 강점
          3) 시장 내 위치
        *결론
        """

        try:
            response = llm.invoke(question)
            return response.content
        except Exception as e:
            return f"OpenAI API 호출 중 오류 발생: {e}"




class SameSectorAnalyzerInputs(BaseModel):
    query: str = Field(..., description="검색하고자 하는 문장")
    company: str = Field(..., description="회사명")
    year: int = Field(..., description="연도")

class SameSectorAnalyzerTool(BaseTool):
    name: str = "SameSectorAnalyzer"
    description: str = _SameSectorComplare_ANL_DESCRIPTION
    args_schema: Type[BaseModel] = SameSectorAnalyzerInputs
    # def _run(self, query: str, metadata: Optional[Dict[str, Any]] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
    def _run(self, query: str, company:str, year:int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:              
            # # 메타데이터 처리
            # filter_dict = {}
            # if metadata and isinstance(metadata, dict):
            #     # companyName 처리
            #     if "companyName" in metadata:
            #         filter_dict["companyName"] = metadata["companyName"]
                
            #     # year 처리
            #     if "year" in metadata:
            #         filter_dict["year"] = metadata["year"]
            filter_dict={}
            filter_dict["year"] = year
            filter_dict["companyName"] = company
            filter_dict["query"] =query
            # SameSectorAnalyzer 초기화
            analyzer = SameSectorAnalyzer(filter_dict["companyName"])

            # 로드에러
            load_error=analyzer.load_competitor()
            if isinstance(load_error, Exception):
                return f"업종 및 순위 정보를 가져오는 중 오류 발생: {load_error}"

            # 재무정보추출에러
            data_error = analyzer.analyze_competitors()
            if isinstance(data_error, Exception):
                return f"기업 데이터 추출 중 오류 발생: {data_error}"

            # 업종 분석 수행 (LLM 사용)
            analysis = analyzer.analyze_sector()
            if isinstance(analysis, Exception):
                return f"업종 분석 중 오류 발생: {analysis}"

            # 결과 문자열 포맷팅
            industry_info = f"업종 분석 결과:\n{analysis}"

            return industry_info
        except Exception as e:
            return f"Unexpected error occurred: {e}"


