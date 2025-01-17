
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import pandas as pd
from typing import Type, Optional
from pydantic import BaseModel, Field
from pydantic import PrivateAttr
from langchain.tools import BaseTool
from selenium.common.exceptions import StaleElementReferenceException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import pandas as pd
from typing import Optional, Dict, Any
from langchain.callbacks.manager import CallbackManagerForToolRun
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool





class NowStockPrice:
    def __init__(self, company_name):
        self.company_name = company_name
        self.stock_price = None

    def setup_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # 브라우저 창 띄우지 않으려면 주석 해제
        self.driver = webdriver.Chrome(options=options)

    def find_company(self):
        try:
            # ChromeDriver 설정 및 네이버 금융 페이지로 이동
            self.setup_driver()
            url = "https://finance.naver.com/"
            self.driver.get(url)

            # 검색창에 회사 이름 입력
            wait = WebDriverWait(self.driver, 10)
            search_input = wait.until(EC.presence_of_element_located((By.ID, "stock_items")))
            search_input.clear()
            search_input.send_keys(self.company_name)

            # 검색 후 첫 번째 결과 클릭
            time.sleep(0.5)  # 잠시 대기
            first_result = self.driver.find_element(By.CSS_SELECTOR, "a._au_real_list")
            first_result.click()

            # 주가 정보 가져오기
            self.get_stock_price()
            return self.stock_price

        except Exception as e:
            return e
        
    def get_stock_price(self):
        max_retries = 3  # 재시도 횟수 제한
        retries = 0
        while retries < max_retries:
            try:
                # WebDriverWait으로 p.no_today 요소 대기
                wait = WebDriverWait(self.driver, 10)
                price_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "p.no_today")))

                # <span> 태그 안의 텍스트 추출
                spans = price_element.find_elements(By.CSS_SELECTOR, "span")
                price_text = "".join(span.text for span in spans if span.text.isdigit() or span.text == ",")  # 숫자와 쉼표만 포함

                # 쉼표 제거 후 숫자로 변환
                self.stock_price = int(price_text.replace(",", ""))

                #print(f"현재 주가: {self.stock_price}원")
                return self.stock_price
            except StaleElementReferenceException:
                #print("stale element 발견, 요소를 다시 찾습니다.")
                retries += 1  # 재시도 횟수 증가
            except Exception as e:
                #print(f"주가 정보 가져오기 중 오류 발생: {e}")
                self.stock_price = None
                return self.stock_price

        #print("재시도 횟수 초과로 주가를 가져오지 못했습니다.")
        self.stock_price = None
        return self.stock_price
    
_STOCKPRICE_DESCRIPTION="""
NowStockPriceTool(company_name:str)->int:
"""

class NowStockPriceInputs(BaseModel):
    company_name: str = Field(..., description="분석할 회사명")


class NowStockPriceTool(BaseTool):
    name: str = "NowStockPrice"
    description: str = _STOCKPRICE_DESCRIPTION
    args_schema: Type[BaseModel] = NowStockPriceInputs

    def _run(self, company_name: str) -> int:
        try:
            stock_tool = NowStockPrice(company_name)
            result = stock_tool.find_company()

            # 결과 반환
            if isinstance(result, int):
                return result
            else:
                return f"주가 정보를 가져오는 데 실패했습니다: {result}"
        except Exception as e:
            return f"Error occurred while fetching stock price: {e}"