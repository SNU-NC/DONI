from langchain_core.prompts import ChatPromptTemplate

class currentQuarterReview:
    """직전 분기 실적 리뷰를 작성"""

    def __init__(self, state, segments, llm):
        self.state = state
        self.segments = segments
        self.llm = llm

    def review(self):
        result_of_business_segment = ""
        for segment in self.segments:
            result_of_business_segment += f"{segment} 사업부의 직전 분기 매출액은 {self.state.segment[segment]['sales']} 입니다. yoy로 {self.state.segment[segment]['yoy']}% 변했습니다.\n"

        result_of_total_business = f"총 매출액은 {self.state.segment['영업수익']['sales']} 입니다. yoy로 {self.state.segment['영업수익']['yoy']}% 변했습니다."

        consensus_of_total_business = f"총 매출액에 대한 컨센서스는 {self.state.segment['영업수익']['sales_consensus']} 였습니다. yoy로 {self.state.segment['영업수익']['yoy_consensus']}% 변화하는 것이 컨센서스였습니다."

        template = """
        당신은 증권사 소속 애널리스트입니다.
        다음 내용을 참고하여 직전 분기 실적에 대한 리뷰를 작성하시오.

        직전 분기 사업부별 매출 실적 : {result_of_business_segment}
        직전 분기 총 매출 실적 : {result_of_total_business}
        직전 분기 총 매출 실적에 대한 컨센서스 : {consensus_of_total_business}

        """
        prompt = ChatPromptTemplate.from_template(template=template)

        chain = prompt | self.llm 
        
        review = chain.invoke({"result_of_business_segment":result_of_business_segment, "result_of_total_business":result_of_total_business, "consensus_of_total_business":consensus_of_total_business})
        return review

