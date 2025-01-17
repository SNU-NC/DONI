from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class yoyPredictionOutput(BaseModel):
    """yoy 예측 아웃풋"""
    business_segment: str = Field(description="사업부")
    yoy: float = Field(description="사업부 yoy 예측값")
    reason: str = Field(description="사업부 yoy 예측값의 근거")

class yoyPrediction:
    def __init__(self, company_name, segment, news, yoy, llm):
        self.company_name = company_name
        self.segment = segment
        self.news = news
        self.yoy = yoy
        self.llm = llm
    
    def predict(self):
        parser = PydanticOutputParser(pydantic_object=yoyPredictionOutput)

        template = """
        당신은 증권사 소속 애널리스트입니다.
        다음 내용을 참고하여 {company_name}의 {business_segment} 사업부의 매출 혹은 비용이 yoy로 얼마나 변할지 예측하시오.

        직전 분기 yoy : {yoy}

        {company_name}의 {business_segment} 사업부의 매출 혹은 비용에 큰 영향을 미치는 뉴스 : {news}

        응답 형식:
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template=template, partial_variables={"format_instructions": parser.get_format_instructions()})

        chain = prompt | self.llm | parser

        result = chain.invoke({"company_name": self.company_name, "news": self.news, "business_segment": self.segment, "yoy": self.yoy})
        return result