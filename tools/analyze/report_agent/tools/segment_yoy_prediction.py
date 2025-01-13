from langchain_core.prompts import ChatPromptTemplate

class segmentYoYpredictionResult:
    def __init__(self, state, segments, llm):
        self.state = state
        self.segments = segments
        self.llm = llm

    def predict(self):
        """사업부별 매출액 yoy 예측결과와 근거를 사업부별로 정리하여 작성"""
        segment_yoy_prediction_result = ""
        for segment in self.segments:
            segment_yoy_prediction_result += f"다음 분기 {segment} 사업부의 매출액은 yoy로 {self.state.segment[segment]['yoy_prediction']}% 변화할 것으로 예측됩니다.\n 직전 분기 {segment} 사업부의 매출액은 {self.state.segment[segment]['yoy']}% 변화했습니다.\n"
            segment_yoy_prediction_result += f"그 이유는 다음과 같습니다. {self.state.segment[segment]['yoy_prediction_reason']}\n"
        segment_yoy_prediction_result += f"이 예측은 다음과 같은 뉴스를 기반으로 이루어졌습니다. {self.state.segment[segment]['news_result']}\n"

        template = """
        당신은 증권사 소속 애널리스트입니다.
        다음 내용을 참고하여 각 사업부별 매출액 예측결과와 근거를 사업부별로 정리하여 작성하세요.
        근거를 서술할 때에는 근거가 된 뉴스를 반드시 언급하세요.
        다음 분기 매출액 yoy 증가율 예측결과와 직전분기 매출액 yoy 증가율을 비교하세요.

        사업부별 매출 예측 결과와 근거 : {segment_yoy_prediction_result}
        """
        prompt = ChatPromptTemplate.from_template(template=template)
        chain = prompt | self.llm
        segment_yoy_prediction_result = chain.invoke({"segment_yoy_prediction_result": segment_yoy_prediction_result}).content
        return segment_yoy_prediction_result