from typing import List, Tuple, Optional, Any, Type
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from sklearn.linear_model import LinearRegression
import numpy as np
from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any
from langchain.callbacks.manager import CallbackManagerForToolRun
import json
import pandas as pd
import os


api_key=os.getenv("OPENAI_API_KEY")
# API 키와 Gateway API 키를 넣습니다.
os.environ["OPENAI_API_KEY"] = api_key

class CompanyAnalyzer:
    def __init__(self, company_name):
        self.company_name = company_name
        self.driver = None
        self.df2 = pd.DataFrame()  # 사전에 "회사명", "EPS증가율평균","PER","ROE평균","PBR","BPS","52주베타값","latest" EPS" 포함한 df 할당 필요
        self.valid_companies = []
        self.last_eps_value = None
        self.predicted_per = None
        self.last_bps_value = None
        self.predicted_pbr = None
        self.existing_stock_names = None #코스피 기업 리스트에 존재하는 회사명
        self.industry_name = None
        self.total_companies = 0
        self.predicted_stock_price=None
        self.predicted_stock_price_pbr=None

    def setup_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)

    def navigate_to_company(self, company_name):
        self.setup_driver()
        url = "https://finance.naver.com/"
        self.driver.get(url)

        wait = WebDriverWait(self.driver, 0.3)
        search_input = wait.until(EC.presence_of_element_located((By.ID, "stock_items")))
        search_input.clear()
        search_input.send_keys(company_name)
        time.sleep(0.7)

        first_result = self.driver.find_element(By.CSS_SELECTOR, "a._au_real_list")
        first_result.click()

    def find_sector_company(self):
        df = pd.read_csv('data/kospi_list.csv')
        self.existing_stock_names = df['종목명'].tolist()
        try:
            #print(f"===== {self.company_name} 기업 업종 정보 조회 시작 =====")
            self.navigate_to_company(self.company_name)
            wait = WebDriverWait(self.driver, 0.3)
            trade_compare_section = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "trade_compare"))
            )
            industry_link = trade_compare_section.find_element(By.CSS_SELECTOR, 'a[href*="/sise/sise_group_detail.naver"]')
            industry_href = industry_link.get_attribute("href")
            self.industry_name = industry_link.text.strip()

            self.driver.get(industry_href)

            industry_table = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'table[summary="업종별 시세 리스트"]'))
            )

            rows = industry_table.find_elements(By.CSS_SELECTOR, "tbody tr")

            for row in rows:
                try:
                    name_area = row.find_element(By.CLASS_NAME, "name_area")
                    company_element = name_area.find_element(By.TAG_NAME, "a")
                    company_text = company_element.text.strip()

                    if company_text in self.existing_stock_names:
                        self.valid_companies.append(company_text) #기업명이 코스피기업 리스트에 있으면 유효한 기업으로 추가
                except Exception:
                    continue

            self.total_companies = len(self.valid_companies)
            if self.total_companies >= 4:
                #print(f"📍업종: {self.industry_name}")
                #print(f"✅ {self.industry_name} 업종에는 {self.total_companies}개의 기업이 포함되어 있습니다.")
                #print(", ".join(self.valid_companies))
                #print("===== 업종 정보 조회 완료 =====\n")
                return self.valid_companies
            else:
                #print(f"❌ {self.industry_name} 업종에 포함된 기업 수가 4개 미만입니다. 작업을 종료합니다.")
                #print("===== 업종 정보 조회 완료 =====\n")
                return "업종에 포함된 기업 수가 4개 미만입니다."
            
        except Exception as e:
            return f"오류 발생: {e}"

        finally:
            if self.driver:
                self.driver.quit()

    def filter_sector_companies(self):
        # 업종 필터링: df2에서 valid_companies에 속하는 기업만 남김
        self.df2=pd.read_csv('data/final_result.csv')
        if not self.valid_companies:
            #print("유효한 업종 기업이 없습니다. 먼저 find_sector_company를 실행해주세요.")
            return "업종의 유효한 기업이 없습니다."

        self.df2 = self.df2[self.df2['회사명'].isin(self.valid_companies)].copy()
        #print("업종 필터링 완료. df2에 업종 내 기업만 남겼습니다.")
        #print(self.df2)

    def regression_eps_beta(self):
        #print("===== EPS증가율평균과 베타값으로 회귀분석 시작 =====")
        if len(self.valid_companies) < 4:
            #print("유효 기업이 4개 미만이므로 회귀분석을 건너뜁니다.")
            return "유효한 기업이 4개 미만이므로 회귀분석을 건너뜁니다"
        if self.df2.empty:
            #print("df2 데이터가 없습니다. 회귀분석을 진행할 수 없습니다.")
            return "df2의 데이터가 비었습니다."
        
        # 기본 필터링
        df_train = self.df2[
            (self.df2['회사명'].str.lower() != self.company_name.lower()) & 
            (self.df2['latest_eps'] > 0)  # 'latest_eps' 값이 양수인 경우만 포함
        ].dropna(subset=['EPS증가율평균', 'PER'])

        # PER 값이 0보다 큰 경우만 포함
        df_train = df_train[df_train['PER'] > 0]

        df_test = self.df2[self.df2['회사명'].str.lower() == self.company_name.lower()].dropna(subset=['EPS증가율평균'])

        # 검색된 회사 데이터에 NaN 값이 포함된 경우 처리
        if df_test.isna().any(axis=None):
            #print(f"'{self.company_name}'의 데이터에 NaN 값이 포함되어 있어 가치평가를 진행할 수 없습니다.")
            return f"'{self.company_name}'의 데이터에 NaN 값이 포함되어 있어 가치평가를 진행할 수 없습니다."

        #initial_df_train = df_train.copy()
        for feature in ['EPS증가율평균', 'PER']:
            Q1 = df_train[feature].quantile(0.25)
            Q3 = df_train[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2 * IQR
            upper_bound = Q3 + 2 * IQR
            df_train = df_train[(df_train[feature] >= lower_bound) & (df_train[feature] <= upper_bound)]

        # excluded_companies = initial_df_train[~initial_df_train.index.isin(df_train.index)]
        # if not excluded_companies.empty:
        #     print("\nEPS와 PER 이상치로 제외된 기업 목록:")
        #     print(excluded_companies[['회사명', 'EPS증가율평균', 'PER']])

        if df_train.empty or df_test.empty:
            return "df_train과 df_test의 데이터가 비어있습니다"

        features = ['EPS증가율평균', '52주베타값']
        X_train = df_train[features].values
        y_train = df_train['PER'].values
        X_test = df_test[features].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_per = model.predict(X_test)
        self.predicted_per=predicted_per
        #print("===== EPS증가율평균과 베타값 회귀분석 완료 =====\n")
        return predicted_per

    def regression_roe_beta(self):
        #print("===== ROE평균과 베타값으로 회귀분석 시작 =====")
        if len(self.valid_companies) < 4:
            return "유효 기업이 4개 미만이므로 회귀분석을 건너뜁니다."
        if self.df2.empty:
            return "df2 데이터가 없습니다. 회귀분석을 진행할 수 없습니다."

        df_train = self.df2[self.df2['회사명'].str.lower() != self.company_name.lower()].dropna(subset=['ROE평균', 'PBR'])
        df_train = df_train[df_train['PBR'] > 0]

        df_test = self.df2[self.df2['회사명'].str.lower() == self.company_name.lower()].dropna(subset=['ROE평균'])
        # 검색된 회사 데이터에 NaN 값이 포함된 경우 처리
        if df_test.isna().any(axis=None):
            return f"'{self.company_name}'의 데이터에 NaN 값이 포함되어 있어 가치평가를 진행할 수 없습니다."


        #initial_df_train = df_train.copy()
        for feature in ['ROE평균', 'PBR']:
            Q1 = df_train[feature].quantile(0.25)
            Q3 = df_train[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2 * IQR
            upper_bound = Q3 + 2 * IQR
            df_train = df_train[(df_train[feature] >= lower_bound) & (df_train[feature] <= upper_bound)]

        # excluded_companies = initial_df_train[~initial_df_train.index.isin(df_train.index)]
        # if not excluded_companies.empty:
        #     print("\nROE와 PBR 이상치로 제외된 기업 목록:")
        #     print(excluded_companies[['회사명', 'ROE평균', 'PBR']])

        if df_train.empty or df_test.empty:
            return "회귀분석에 필요한 데이터가 부족합니다"

        features = ['ROE평균', '52주베타값']
        X_train = df_train[features].values
        y_train = df_train['PBR'].values
        X_test = df_test[features].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_pbr = model.predict(X_test)
        self.predicted_pbr = predicted_pbr
        return predicted_pbr

    
        #print("===== ROE평균과 베타값 회귀분석 완료 =====\n")

    def stock_price_prediction(self):
        if len(self.valid_companies) < 4:
            return "유효 기업이 4개 미만이므로 주가 예측을 건너뜁니다."
        
        company_row = self.df2[self.df2['회사명'] == self.company_name]
        # NaN 값 확인 및 처리
        if company_row.isna().any(axis=None):  # 데이터프레임에 NaN 값이 하나라도 있는 경우
            return f"'{self.company_name}'의 데이터는 재무제표의 지표들이 주어지지 않아 가치평가를 진행할 수 없습니다."

        message = ""  # 상태 메시지를 저장할 변수

        # PER 기반 주가
        self.last_eps_value = company_row['latest_eps'].values
        if self.last_eps_value.size == 0:
            message += "가장 최근 EPS 값이 없어서 PER 기반 가치평가를 진행할 수 없습니다.\n"
        elif self.predicted_per:
            try:
                last_eps = self.last_eps_value[-1]
                predicted_per = self.predicted_per[0]
                if predicted_per < 0:
                    message += f"📉 예측 PER 값이 음수(-{abs(predicted_per):.2f})입니다. 따라서 PER 기반 주가를 도출할 수 없습니다.\n"
                elif last_eps < 0:
                    message += "🚫 가장 최근 분기의 EPS 값이 음수이므로 PER 기반 기업 가치를 평가할 수 없습니다.\n"
                else:
                    self.predicted_stock_price = last_eps * predicted_per
                    message += f"📝 {self.company_name}의 PER 기반 가치평가 결과: {self.predicted_stock_price:.0f}원.\n"
            except (IndexError, TypeError) as e:
                message += f"PER 기반 주가 계산 중 오류 발생: {e}\n"
        else:
            message += "예측 PER 값이 없어 PER 기반 주가 예측을 건너뜁니다.\n"

        # PBR 기반 주가
        self.last_bps_value = company_row['BPS'].values
        if self.predicted_pbr is not None:
            try:
                predicted_pbr = self.predicted_pbr[0]
                if predicted_pbr < 0:
                    message += f"📉 예측 PBR 값이 음수(-{abs(predicted_pbr):.2f})입니다. 따라서 PBR 기반 주가를 도출할 수 없습니다.\n"
                elif self.last_bps_value.size == 0:
                    message += "🚫 BPS 값이 없어 PBR 기반 주가 예측을 진행할 수 없습니다.\n"
                else:
                    self.predicted_stock_price_pbr = self.last_bps_value[-1] * predicted_pbr
                    message += f"📝 {self.company_name}의 PBR 기반 가치평가 결과: {self.predicted_stock_price_pbr:.0f}원.\n"
            except (IndexError, TypeError) as e:
                message += f"PBR 기반 주가 계산 중 오류 발생: {e}\n"
        else:
            message += "예측 PBR 값이 없어 PBR 기반 주가 예측을 건너뜁니다.\n"

        return message.strip()





        

        

_ANL_DESCRIPTION = """
CompanyAnalyzer(company_name:str) -> str:
"""

class CompanyAnalyzerInputs(BaseModel):
    company_name: str = Field(..., description="분석할 회사명")



class CompanyAnalyzerTool(BaseTool):
    name: str = "CompanyAnalyzer"
    description: str = _ANL_DESCRIPTION
    args_schema: Type[BaseModel] = CompanyAnalyzerInputs

    def _run(self, company_name: str) -> str:
        try:
            # CompanyAnalyzer 초기화
            analyzer = CompanyAnalyzer(company_name)

            # Navigate to company
            navigation_result = analyzer.navigate_to_company(company_name)
            if isinstance(navigation_result, str):  # Error string returned
                return navigation_result

            # Find sector companies
            sector_result = analyzer.find_sector_company()
            if isinstance(sector_result, str):  # Error string returned
                return sector_result

            # Filter sector companies
            filter_result = analyzer.filter_sector_companies()
            if isinstance(filter_result, str):  # Error string returned
                return filter_result

            # Predict PER
            predicted_per = analyzer.regression_eps_beta()
            if isinstance(predicted_per, str):  # Error string returned
                return predicted_per

            # Predict PBR
            predicted_pbr = analyzer.regression_roe_beta()
            if isinstance(predicted_pbr, str):  # Error string returned
                return predicted_pbr

            # Predict stock prices
            stock_price_result = analyzer.stock_price_prediction()
            if isinstance(stock_price_result, str):  # Error string returned
                return stock_price_result

            # Prepare the final result if everything is successful
            predicted_stock_price_per, predicted_stock_price_pbr = stock_price_result
            result = (
                f"분석 결과:\n"
                f"입력된 기업명: {company_name}\n"
                f"predicted_per: {predicted_per}\n"
                f"predicted_pbr: {predicted_pbr}\n"
                f"predicted_stock_price_per: {predicted_stock_price_per}\n"
                f"predicted_stock_price_pbr: {predicted_stock_price_pbr}"
            )
            return result

        except Exception as e:
            return f"Unexpected error occurred: {e}"
        
            