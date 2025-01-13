from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from dotenv import load_dotenv
from selenium.common.exceptions import TimeoutException
import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
import pandas as pd
import os
import time
import re
from langchain.callbacks.manager import CallbackManagerForToolRun
from config.prompts import _STOCK_ANL_DESCRIPTION
api_key=os.getenv("OPENAI_API_KEY")
# API 키와 Gateway API 키를 넣습니다.
os.environ["OPENAI_API_KEY"] = api_key





class StockAnalyzer:
    def __init__(self, api_key=None):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API 키를 찾을 수 없습니다.")

    def clean_rate(self, rate):
        try:
            rate = re.sub(r'[^\d.-]', '', rate)
            return float(rate)
        except ValueError:
            #print(f"등락률 변환 실패: {rate}")
            return None

    def get_stock_data(self, corpname):
        #print("주식 데이터 수집 시작")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        

        try:
            driver.get("https://tossinvest.com/")
            wait = WebDriverWait(driver, 5)

            # 검색 입력
            search_input = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'span.tw-1r5dc8g0'))
            )
            search_input.click()

            search_box = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'input[placeholder="검색어를 입력해주세요"]')
                )
            )
            search_box.clear()
            search_box.send_keys(corpname)

            company_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='_1afau9j2']"))
            )
            company_button.click()

            time.sleep(1)
            daily_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, '//button[@value="일별"]'))
            )
            daily_button.click()

            # 데이터 수집
            data = []
            collected_indices = set()

            while len(data) < 30: #### 살펴볼 날짜 수
                rows = driver.find_elements(By.XPATH, '//tr[@data-item-index]')

                for row in rows:
                    try:
                        data_index = int(row.get_attribute('data-item-index'))
                        if data_index in collected_indices:
                            continue

                        cells = row.find_elements(By.TAG_NAME,'td')
                        if len(cells) >= 3:
                            date = cells[0].text
                            closing_price = cells[1].text
                            change_rate = cells[2].text
                            data.append({
                                "날짜": date,
                                "종가": closing_price,
                                "등락률": change_rate
                            })
                            collected_indices.add(data_index)
                        driver.execute_script("arguments[0].scrollIntoView();", row)
                        time.sleep(0.05)
                    except Exception as e:
                        return e
                if len(data) >= 30: #날짜범위설정
                    break
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

            if not data:
                #print("데이터 수집 실패: 데이터가 비어있습니다.")
                return None

            df = pd.DataFrame(data)
            #print("📍수집된 데이터 예시 앞부분:", df)
            # 등락률 데이터를 숫자로 변환하고 NaN 제거
            df["등락률"] = df["등락률"].apply(self.clean_rate)  # 문자열을 숫자로 변환
            df = df.dropna(subset=["등락률"])  # NaN 값 제거

            if df.empty:
                print("데이터프레임이 비어있습니다.")
                return None
            # 절대값 열 추가
            df["등락률_절대값"] = df["등락률"].abs()
            # 양수 및 음수 등락률 최대값의 날짜 추출
            positive_max_date = None
            negative_max_date = None

            positive_data = df[df["등락률"] > 0]
            negative_data = df[df["등락률"] < 0]

            if not positive_data.empty:
                positive_max_date = positive_data.loc[positive_data["등락률"].idxmax(), "날짜"]

            if not negative_data.empty:
                negative_max_date = negative_data.loc[negative_data["등락률"].idxmin(), "날짜"]


            #print("스크롤 가능한 div 로드 대기 중...")
            scrollable_div = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[style*='overflow: auto']"))
            )

            # 투자자 데이터 추출
            personal_positive, foreign_positive, institution_positive = "N/A", "N/A", "N/A"
            personal_negative, foreign_negative, institution_negative = "N/A", "N/A", "N/A"

            last_row_count = 0
            scroll_step = 50

            while True:
                rows = scrollable_div.find_elements(By.CSS_SELECTOR, "tr.tw-mm1vn12.f2ww7n1._1p5yqoh0")
                #print(f"현재 {len(rows)}개의 행을 탐색 중...")

                for index, row in enumerate(rows):
                    try:
                        date_element = row.find_element(
                            By.CSS_SELECTOR, "div.tw-kywygh2.tw-1h28joa4.tw-kywygh5"
                        )
                        date_text = date_element.text.strip()

                        # 양수 최대 날짜 처리
                        if positive_max_date and positive_max_date in date_text:
                            #print(f"양수 최대 날짜 '{positive_max_date}'와 일치하는 데이터를 찾았습니다.")
                            personal_positive = row.find_element(By.CSS_SELECTOR, "td:nth-child(2) span.tw-1r5dc8g0").text.strip()
                            foreign_positive = row.find_element(By.CSS_SELECTOR, "td:nth-child(3) span.tw-1r5dc8g0").text.strip()
                            institution_positive = row.find_element(By.CSS_SELECTOR, "td:nth-child(4) span.tw-1r5dc8g0").text.strip()

                        # 음수 최대 날짜 처리
                        if negative_max_date and negative_max_date in date_text:
                            #print(f"음수 최대 날짜 '{negative_max_date}'와 일치하는 데이터를 찾았습니다.")
                            personal_negative = row.find_element(By.CSS_SELECTOR, "td:nth-child(2) span.tw-1r5dc8g0").text.strip()
                            foreign_negative = row.find_element(By.CSS_SELECTOR, "td:nth-child(3) span.tw-1r5dc8g0").text.strip()
                            institution_negative = row.find_element(By.CSS_SELECTOR, "td:nth-child(4) span.tw-1r5dc8g0").text.strip()

                        # 두 날짜 정보가 모두 수집되었으면 종료
                        if positive_max_date and negative_max_date and \
                        personal_positive != "N/A" and personal_negative != "N/A":
                            break

                    except Exception as e:
                        #print(f"행 {index + 1}에서 예외 발생: {e}")
                        continue

                driver.execute_script("arguments[0].scrollTop += arguments[1];", scrollable_div, scroll_step)
                time.sleep(1)

                new_rows = scrollable_div.find_elements(By.CSS_SELECTOR, "tr.tw-mm1vn12.f2ww7n1._1p5yqoh0")
                if len(new_rows) == last_row_count:
                    #print("더 이상 로드할 데이터가 없습니다.")
                    break
                last_row_count = len(new_rows)

            return {
                "양수_최대": {
                    "날짜": positive_max_date,
                    "개인": personal_positive,
                    "외국인": foreign_positive,
                    "기관": institution_positive,
                    "등락률": df.loc[df["날짜"] == positive_max_date, "등락률"].values[0]
                } if positive_max_date else None,
                "음수_최대": {
                    "날짜": negative_max_date,
                    "개인": personal_negative,
                    "외국인": foreign_negative,
                    "기관": institution_negative,
                    "등락률": df.loc[df["날짜"] == negative_max_date, "등락률"].values[0]
                } if negative_max_date else None
            }

        except Exception as e:
            #print(f"주식 데이터 수집 중 오류 발생: {e}")
            return None

        finally:
            driver.quit()

    def get_news_analysis(self, corpname, stock_data):
        #print("뉴스 분석 시작")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)

        try:
            analysis_results = {}

            for key, stock_info in {"양수_최대": stock_data.get("양수_최대"), "음수_최대": stock_data.get("음수_최대")}.items():
                if not stock_info:
                    analysis_results[key] = "해당 조건의 날짜에 대한 데이터가 없습니다."
                    continue

                # 각 조건에 맞는 날짜 설정
                target_date = stock_info['날짜']
                #print(f"{key}의 검색 날짜: {target_date}")  # 디버깅용 날짜 확인

                # URL 생성
                news_url = (
                    f"https://search.naver.com/search.naver?"
                    f"where=news&query={corpname}&sm=tab_opt&sort=0&photo=0&field=0"
                    f"&pd=3&ds=2024.{target_date}&de=2024.{target_date}&docid=&related=0"
                    f"&mynews=0&office_type=0&office_section_code=0&news_office_checked="
                    f"&nso=so%3Ar%2Cp%3Afrom{target_date.replace('.', '')}to{target_date.replace('.', '')}"
                    f"&is_sug_officeid=0&office_category=0&service_area=0"
                )

                # 뉴스 검색
                driver.get(news_url)

                wait = WebDriverWait(driver, 5)
                try:
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.news_tit')))
                except TimeoutException:
                    analysis_results[key] = "해당 날짜에 대한 뉴스 데이터가 없습니다."
                    continue
                news_elements = driver.find_elements(By.CSS_SELECTOR, 'a.news_tit')

                # 뉴스 제목 수집
                titles = [news.text for news in news_elements[:15]]
                #print(f"{key} 기준 수집된 뉴스 제목:", titles)

                llm = ChatOpenAI(
                    model_name="gpt-4o",
                    openai_api_key=self.api_key
                )

                question = (
                    f"다음 뉴스 제목들 {titles}을 바탕으로 {corpname} 주식이 변동한 이유를 분석해주세요. "
                    f"해당 날짜({target_date}) 기준 개인투자자: {stock_info['개인']}, "
                    f"외국인투자자: {stock_info['외국인']}, 기관투자자: {stock_info['기관']}의 투자 동향도 함께 고려해주세요. "
                    f"등락률: {stock_info['등락률']}이 {'양수' if key == '양수_최대' else '음수'}면 "
                    f"{'긍정적인' if key == '양수_최대' else '부정적인'} 이슈를 중심으로 분석해주세요."
                )

                response = llm(question)
                analysis_results[key] = response.content

            return analysis_results

        except Exception as e:
            #print(f"뉴스 분석 중 오류 발생: {e}")
            return None

        finally:
            driver.quit()

    def analyze_stock(self, corpname):
        # 전체 주식 분석 시작
        stock_data = self.get_stock_data(corpname)
        if not stock_data:
            #print("주식 데이터를 수집할 수 없습니다.")
            return None

        news_analysis = self.get_news_analysis(corpname, stock_data)
        return {
            "양수_최대_분석": news_analysis.get("양수_최대"),
            "음수_최대_분석": news_analysis.get("음수_최대"),
            "주식_데이터": stock_data
        }


class StockAnalyzerInputs(BaseModel):
    query: str = Field(..., description="검색하고자 하는 문장")
    company: str = Field(..., description="회사명")
    year: int = Field(..., description="연도")

class StockAnalyzerTool(BaseTool):
    name: str = "StockAnalyzerTool"
    description: str = _STOCK_ANL_DESCRIPTION
    args_schema: Type[BaseModel] = StockAnalyzerInputs

    def _run(self, query: str, company:str, year:int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:              
            filter_dict={}
            filter_dict["year"] = year
            filter_dict["companyName"] = company
            filter_dict["query"] =query
                    
            # StockAnalyzer 실행
            analyzer = StockAnalyzer()
            result = analyzer.analyze_stock(filter_dict["companyName"])

            if not result:
                return "주식 데이터를 수집하거나 분석하는데 실패했습니다."

            stock_data = result.get('주식_데이터', {})
            news_analysis = {
                "양수_최대": result.get('양수_최대_분석', "양수 최대 등락률 날짜에 대한 뉴스 분석 결과가 없습니다."),
                "음수_최대": result.get('음수_최대_분석', "음수 최대 등락률 날짜에 대한 뉴스 분석 결과가 없습니다.")
            }

            # 주식 데이터가 없을 경우 처리
            if not stock_data:
                return "주식 데이터를 수집할 수 없습니다."

            # 양수 최대 및 음수 최대 주식 데이터 문자열 포맷팅
            positive_stock_data = stock_data.get("양수_최대", None)
            negative_stock_data = stock_data.get("음수_최대", None)

            positive_data_str = (
                f"=== 양수 최대 등락률 ===\n"
                f"날짜: {positive_stock_data.get('날짜', 'N/A')}\n"
                f"개인: {positive_stock_data.get('개인', 'N/A')}\n"
                f"외국인: {positive_stock_data.get('외국인', 'N/A')}\n"
                f"기관: {positive_stock_data.get('기관', 'N/A')}\n"
                f"등락률: {positive_stock_data.get('등락률', 'N/A')}\n"
            ) if positive_stock_data else "양수 최대 등락률 데이터가 없습니다."

            negative_data_str = (
                f"=== 음수 최대 등락률 ===\n"
                f"날짜: {negative_stock_data.get('날짜', 'N/A')}\n"
                f"개인: {negative_stock_data.get('개인', 'N/A')}\n"
                f"외국인: {negative_stock_data.get('외국인', 'N/A')}\n"
                f"기관: {negative_stock_data.get('기관', 'N/A')}\n"
                f"등락률: {negative_stock_data.get('등락률', 'N/A')}\n"
            ) if negative_stock_data else "음수 최대 등락률 데이터가 없습니다."

            # 최종 결과 문자열 조합
            final_str = (
                f"매칭된 기업명: {filter_dict['companyName']}\n\n"
                f"{positive_data_str}\n"
                f"=== 양수 최대 뉴스 분석 ===\n{news_analysis['양수_최대']}\n\n"
                f"{negative_data_str}\n"
                f"=== 음수 최대 뉴스 분석 ===\n{news_analysis['음수_최대']}\n"
            )

            return final_str
        except Exception as e:
            return f"Unexpected error occurred: {e}"