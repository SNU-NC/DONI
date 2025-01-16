import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

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
        print(f"사용중인 LLM: {type(self.llm).__name__}")    
    def _parse_clova_response(self, response: str) -> Dict[str, Any]:
        """하이퍼클로바 응답을 파싱하는 메서드"""
        try:
            # 1. JSON 부분만 추출
            import re
            json_match = re.search(r'\{[^{]*\}', response)
            if not json_match:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # 2. 필수 필드 검증
            required_fields = ['business_segment', 'yoy', 'reason']
            if not all(field in parsed for field in required_fields):
                raise ValueError("필수 필드가 누락되었습니다")
            
            # 3. yoy 타입 변환 및 검증
            try:
                if isinstance(parsed['yoy'], str):
                    # 문자열에서 숫자만 추출 (예: "10%" -> 10)
                    parsed['yoy'] = float(re.sub(r'[^0-9.-]', '', parsed['yoy']))
            except (ValueError, TypeError):
                logging.warning(f"yoy 값 변환 실패: {parsed['yoy']}")
                parsed['yoy'] = 0.0
            
            return parsed
            
        except Exception as e:
            logging.error(f"응답 파싱 오류: {str(e)}")
            return {
                "business_segment": self.segment,
                "yoy": 0.0,
                "reason": "데이터 분석 중 오류가 발생했습니다"
            }

    def predict(self):
        try:
            # 1. 프롬프트 생성 로깅
            print(" primpt start")
            parser = PydanticOutputParser(pydantic_object=yoyPredictionOutput)
            format_str = """{{
                "business_segment": "사업부명",
                "yoy": "YoY 예측값(숫자)",
                "reason": "예측 근거"
            }}"""

            template = """
            당신은 증권사 소속 애널리스트입니다.
            다음 내용을 참고하여 {company_name}의 {segment} 사업부의 매출 혹은 비용이 yoy로 얼마나 변할지 예측하시오.

            직전 분기 yoy : {yoy}

            {company_name}의 {segment} 사업부의 매출 혹은 비용에 큰 영향을 미치는 뉴스 : {news}

            주의사항:
            1. 추가 설명 없이 JSON만 반환하세요
            2. yoy는 반드시 숫자로 입력하세요 (예: 10.5)
            3. 다음 형식을 정확히 따르세요:

            {format_str}
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", template)
            ])

            # 2. LLM 호출 및 응답 로깅
            logging.info("LLM 호출 시작")
            response = self.llm.invoke(prompt.format(
                company_name=self.company_name,
                segment=self.segment,
                yoy=self.yoy,
                news=self.news,
                format_str=format_str
            ))
            print("8번 예측진행 결과", response)
            logging.info(f"LLM 응답 타입: {type(response)}")
            logging.info(f"LLM 응답 내용: {response}")

            # 3. 응답 파싱
            if hasattr(response, 'content'):
                # AIMessage인 경우
                response_text = response.content
                logging.info(f"응답 content 추출: {response_text}")
            else:
                # 직접 문자열인 경우
                response_text = str(response)
                logging.info(f"응답 문자열 변환: {response_text}")

            # 4. JSON 파싱
            try:
                parsed_response = json.loads(response_text)
                logging.info(f"JSON 파싱 결과: {parsed_response}")
                
                # yoy 값 변환
                if isinstance(parsed_response['yoy'], str):
                    parsed_response['yoy'] = float(parsed_response['yoy'].strip('%'))
                    logging.info(f"yoy 값 변환 완료: {parsed_response['yoy']}")
                
                return yoyPredictionOutput(**parsed_response)
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 실패: {e}")
                raise
            
        except Exception as e:
            logging.error(f"예측 중 오류 발생: {str(e)}")
            # 기본값 반환
            return yoyPredictionOutput(
                business_segment=self.segment,
                yoy=0.0,
                reason="예측 중 오류가 발생했습니다"
            )