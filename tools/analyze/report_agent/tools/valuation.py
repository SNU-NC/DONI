import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate

class Valuation:
    def __init__(self, state, income_stmt_cum, ticker, llm):
        self.state = state
        self.income_stmt_cum = income_stmt_cum
        self.ticker = ticker
        self.llm = llm

    def estimate(self):
        """목표 주가 산정"""
        ticker = yf.Ticker(self.ticker+".KS")
        estEarnings = self.income_stmt_cum.loc[self.income_stmt_cum['계정'] == '순이익','next_quarter'].values[0]
        shares = ticker.info.get("sharesOutstanding")
        estEPS = estEarnings/shares
        price_consensus = ticker.analyst_price_targets

        template = """
        당신은 증권사 소속 애널리스트입니다.
        다음 내용을 참고하여 {company_name}의 목표 주가를 설정하고 그 이유를 반드시 서술하세요.
        직전분기 매출 실적 리뷰 내용과 사업부별 매출 예측 결과와 근거는 반드시 서술하세요.
        목표 주가를 설정할 때에는 먼저, 현재 기업 PER과 경쟁사들의 평균 PER을 반드시 고려하여 기업의 목표 PER을 설정하세요.
        목표 PER과 추정 EPS를 곱하여 목표 주가를 설정하세요.

        직전 분기 매출 실적 리뷰: {current_quarter_review_result}

        사업부별 매출 예측 결과와 근거 : {segment_yoy_prediction_result}

        현재 기업 PER : {PER}

        기업 경쟁사 리스트 : {peer_list}

        현재 경쟁사들의 평균 PER : {average_peer_PER}

        추정 EPS : {EPS} (추정 매출액을 기반으로 계산된 결과입니다.)

        목표 주가를 제안한 이후, 목표 주가 컨센서스를 아래 내용을 참고하여 작성하세요.

        평균 목표 주가 : {avg_price_consensus}
        최소 목표 주가 : {low_price_consensus}
        최대 목표 주가 : {high_price_consensus}
        현재 주가 : {current_price}
        """


        prompt = ChatPromptTemplate.from_template(template=template)
        chain = prompt | self.llm
        valuation_result = chain.invoke({"company_name": self.state.company_name, 
                    "current_quarter_review_result": self.state.result['sales_review'], 
                    "segment_yoy_prediction_result": self.state.result['segment_yoy_prediction_result'], 
                    "PER": self.state.PER, 
                    "average_peer_PER": self.state.average_peer_PER,
                    "peer_list": self.state.peer_list,
                    "EPS": estEPS,
                    "avg_price_consensus": price_consensus['mean'],
                    "low_price_consensus": price_consensus['low'],
                    "high_price_consensus": price_consensus['high'],
                    "current_price": ticker.info.get("regularMarketPrice")})
        return valuation_result