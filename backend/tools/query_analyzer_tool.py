from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# dotenv 로드
load_dotenv()

llm_mini = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

def query_analyzer(query: str) -> Dict[str, bool]:
    """
    사용자 쿼리를 분석하여 quick_retriever_tool과 plan_and_schedule 사용 여부를 결정

    Args:
        query (str): 사용자 쿼리
        llm (BaseChatModel): 언어 모델

    Returns:
        Dict[str, bool]: quick_retriever_tool과 plan_and_schedule 사용 여부
    """
    financial_terms = [
        # 0. 돈
        "영업이익", "매출액", "자산총계", "부채총계", "자본총계",

        # 1. 안정성비율 
        "유동비율", "부채비율", "이자보상배율", "자기자본비율", "판매비와관리비", "판관비",
        
        # 2. 성장성비율, 수익성비율
        "EPS", "영업이익률", "EBITDA", "ROA", "ROE", "ROIC",
        
        # 3. 활동성비율
        "총자산회전율"
    ]

    prompt = f"""
    당신은 사용자의 질문에 대해 두 도구의 사용 여부를 판단하는 전문가 입니다.
    반드시 True 또는 False로만 답변하되, 아래 형식을 정확히 지켜주세요.

    아래는 두 도구에 대한 설명입니다:
    1. quick_retriever_tool
    - FnGuide에서 단순 재무지표를 조회하는 도구
    - FnGuide에서 조회 가능한 재무지표는 다음과 같습니다: {financial_terms}
    - FnGuide에서 "분기실적"과 같은 상세 정보는 얻기 어렵습니다.

    2. plan_and_schedule
    - FnGuide에서 찾을 수 없는 재무지표를 조회하는 도구
        - 예시: ["당기순이익", "영업수익", "현금및현금성자산", "운용수익", "영업활동현금흐름", "배당수익률", "배당성향"]
    - 사업보고서 검색이나 웹검색, 애널리스트 보고서 투자의견 등이 필요한 경우 사용하는 도구
    - 분기 실적에 대한 상세한 조회가 필요한 경우 사용하는 경우
    - 평균, 변화율 계산 등 재무지표들의 계산이 필요한 경우 사용하는 도구

    아래는 답변 예시입니다:
    "삼성전자 영업이익이 얼마야?" -> {{"quick_retriever_tool": True, "plan_and_schedule": False}}
    "카카오의 주요 매출원은 무엇인가요?" -> {{"quick_retriever_tool": False, "plan_and_schedule": True}}
    "GS리테일의 2023년 당기순이익은?" -> {{"quick_retriever_tool": False, "plan_and_schedule": True}}
    "LG화학의 분기 실적(매출, 영업이익, 순이익)을 상세히 알려주세요." -> {{"quick_retriever_tool": False, "plan_and_schedule": True}}

    사용자 질문: "{query}"

    출력은 다음 JSON 스키마를 반드시 정확히 따라야 합니다:
    {{"quick_retriever_tool": True/False, "plan_and_schedule": True/False}}
    """

    try:
        response = llm_mini.invoke(prompt)
        result = eval(response.content.strip())  # 문자열을 dict로 변환
        return result
    except Exception as e:
        print(f"Error in analyze_tools_to_use: {e}")
        # 오류 발생시 기본값으로 plan_and_schedule만 사용
        return {"quick_retriever": False, "plan_and_schedule": True}