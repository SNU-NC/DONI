from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from tools.analyze.report_agent.tools.report_agent_utils import get_ticker, extract_segment, yoy_calculator, consensusCalculator, combine_report
from tools.analyze.report_agent.tools.yoy_prediction import yoyPrediction
from tools.analyze.report_agent.tools.predict_next_qt import predictNextQuarter
from tools.analyze.report_agent.tools.current_qt_review import currentQuarterReview
from tools.analyze.report_agent.tools.find_per import find_peer_PERs_tool, find_PER_tool
from tools.analyze.report_agent.tools.segment_yoy_prediction import segmentYoYpredictionResult
from tools.analyze.report_agent.tools.valuation import Valuation
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import ToolException
import pandas as pd
import logging
import os
import time

# 상태 정보 정의

class State(BaseModel):
    company_name: str = Field(description="회사 이름", default="")
    ticker: str = Field(description="회사 티커", default="")
    segment: Dict[str, dict] = Field(description="사업부 정보", default_factory=dict)
    PER: float = Field(description="기업의 현재 PER(TTM)", default=0)
    peer_list: List[str] = Field(description="경쟁사 리스트", default_factory=list)
    average_peer_PER: float = Field(description="경쟁사 PER 평균", default=0)
    result: Dict[str, str] = Field(description="결과", default_factory=dict)

class ReportAgentManager:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.consensus_df = pd.read_csv("tools/analyze/report_agent/tools/data/consensus_result.csv")
        self.state = State()
        # 프로젝트 루트 디렉토리 설정
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.project_root, 'output', 'predicted_quarterly_financial_data')
        os.makedirs(self.output_dir, exist_ok=True)

    def get_report(self, query: str, filter: Optional[Dict[str, Any]] = None) -> str:
        try:
            print(f"\n=== 리포트 에이전트 프로세스 시작 ===")
            print(f"입력 쿼리: {query}")
            print(f"필터 조건: {filter}")

            # 1. ticker 추출
            self.state.company_name = filter["companyName"]
            # 현대자동차 티커 예외처리
            if self.state.company_name == "현대자동차":
                self.state.company_name = "현대차"
            ticker = self.consensus_df.loc[self.consensus_df['종목명'] == self.state.company_name, '종목코드'].values[0]
            if ticker is None:
                error_msg = f"티커를 찾을 수 없습니다: {self.state.company_name}"
                self.logger.error(error_msg)
                return error_msg
            self.state.ticker = ticker

            print(f"회사명: {self.state.company_name}")
            print(f"티커: {self.state.ticker}")
            print("===1. 회사명 및 티커 추출 완료===")

            # 2. 분기별 요약손익계산서(+사업부별매출) 가져오기
            try:
                file_path = f"./tools/analyze/report_agent/tools/data/quarterly_financial_data/{self.state.ticker}_quarterly_financial_data.xlsx"
                if not os.path.exists(file_path):
                    error_msg = f"분기별 요약손익계산서 파일을 찾을 수 없습니다: {file_path}"
                    self.logger.error(error_msg)
                    return error_msg
                
                self.income_stmt_cum = pd.read_excel(file_path, header=0)
                self.segments = extract_segment(self.income_stmt_cum)
                print("===2. 분기별 요약손익계산서 및 사업부별 매출액 추출 완료===")
            except Exception as e:
                error_msg = f"분기별 요약손익계산서 처리 중 오류 발생: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

            # 3. 직전분기 매출액 컨센서스 가져오기
            current_quarter_sales_consensus = self.consensus_df.loc[self.consensus_df['종목코드'] == self.state.ticker, '직전분기_매출액_컨센서스'].values[0] * (10 ** 8) #(10^8 =억원)
            print("===3. 직전분기 매출액 컨센서스 추출 완료===")

            # 4. 직전분기 누적 매출액 컨센서스 및 yoy 컨센서스 계산
            consensus_calculator = consensusCalculator(self.income_stmt_cum, current_quarter_sales_consensus, self.state)
            yoy_consensus, current_quarter_sales_cum_consensus = consensus_calculator.calculate()
            if '영업수익' not in self.state.segment:
                self.state.segment['영업수익'] = {}
            self.state.segment['영업수익'].update({"yoy_consensus":yoy_consensus})
            self.state.segment['영업수익'].update({"sales_consensus":current_quarter_sales_cum_consensus})
            print("===4. 직전분기 누적 매출액 컨센서스 및 yoy 컨센서스 계산 완료===")

            # 5. 사업부별 YOY 계산
            yoy_series = yoy_calculator(self.income_stmt_cum)
            for segment, yoy in zip(self.income_stmt_cum.iloc[:, 0], yoy_series):
                if segment in self.segments:
                    self.state.segment[segment] = {"yoy": yoy}
                if segment == '영업수익':
                    self.state.segment['영업수익'].update({"yoy": yoy})
            print("===5. 사업부별 YOY 계산 완료===")

            # 6. 사업부별 매출액(영업수익) 추출
            present_quarter = self.income_stmt_cum.iloc[:,1+4]
            for segment, sales in zip(self.income_stmt_cum['계정'], present_quarter):
                if segment in self.segments:
                    self.state.segment[segment].update({"sales":sales})
                if segment == '영업수익':
                    self.state.segment['영업수익'].update({"sales":sales})
            print("===6. 사업부별 매출액(영업수익) 추출 완료===")

            # 7. 직전분기 실적 리뷰 작성
            current_quarter_review = currentQuarterReview(self.state, self.segments, self.llm)
            review = current_quarter_review.review()
            self.state.result.update({"sales_review":review.content})
            print("===7. 직전분기 실적 리뷰 작성 완료===")

            # 8. 사업부별 뉴스 검색
            duckduckgo_search_tool = DuckDuckGoSearchResults(max_results=1,backend="news")
            for segment in self.segments:
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        news_result = duckduckgo_search_tool.invoke(
                            {
                                "query": f"{self.state.company_name} {segment} 사업부 매출"
                            }
                        )
                        self.state.segment[segment].update({"news_result":news_result})
                        break  # 성공시 루프 종료
                    except Exception as e:
                        retry_count += 1
                        self.logger.warning(f"{segment} 사업부 뉴스 검색 {retry_count}번째 실패: {str(e)}")
                        if retry_count == max_retries:
                            self.state.segment[segment].update({"news_result": []})
                        else:
                            time.sleep(1)
            print("===8. 사업부별 뉴스 검색 완료===")

            # 9. 사업부별 YOY 예측
            for segment in self.segments:
                yoy_prediction = yoyPrediction(self.state.company_name, segment, self.state.segment[segment]['news_result'], self.state.segment[segment]['yoy'], self.llm)
                result = yoy_prediction.predict()
                # 예측값 및 예측 이유업데이트
                self.state.segment[segment].update({"yoy_prediction":result.yoy})
                self.state.segment[segment].update({"yoy_prediction_reason":result.reason})
            print("===9. 사업부별 YOY 예측 완료===")

            # 10. 다음분기 사업부별 매출액 yoy 예측 결과와 근거 작성
            segment_yoy_prediction_result = segmentYoYpredictionResult(self.state, self.segments, self.llm)
            result = segment_yoy_prediction_result.predict()
            self.state.result.update({"segment_yoy_prediction_result":result})
            print("===10. 다음분기 사업부별 매출액 yoy 예측 결과와 근거 작성 완료===")

            # 11. 다음 분기 매출액, 영업이익, 순이익 예측
            predict_next_qt = predictNextQuarter(self.income_stmt_cum, self.state, self.segments)
            result_df = predict_next_qt.fill_next_quarter_df()
            self.income_stmt_cum = result_df
            print("===11. 다음 분기 매출액, 영업이익, 순이익 예측 완료===")

            # 12. 목표 PER 산정을 위한 peer PER 및 현재 PER 확인
            self.state.PER = find_PER_tool(self.state.ticker)
            self.state.peer_list, self.state.average_peer_PER = find_peer_PERs_tool(self.state.company_name, self.state, self.llm)
            print("===12. 목표 PER 산정을 위한 peer PER 및 현재 PER 확인 완료===")

            # 13. 밸류에이션(목표 주가 산정)
            valuation = Valuation(self.state, self.income_stmt_cum, self.state.ticker, self.llm)
            result = valuation.estimate()
            self.state.result.update({"valuation_result":result.content})
            print("===13. 밸류에이션(목표 주가 산정) 완료===")

            # 14. 리포트 최종 출력
            report = combine_report(self.state)
            print("===14. 리포트 최종 출력 완료===")
            return report

        except Exception as e:
            error_msg = f"리포트 에이전트 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}"
            self.logger.error(error_msg)
            raise ToolException(error_msg)
