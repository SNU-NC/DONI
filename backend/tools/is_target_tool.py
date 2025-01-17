import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# dotenv 로드
load_dotenv()

llm_mini = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

def is_valid_query(query: str) -> bool:
    """
    Args:
        query (str): 사용자 쿼리 입력
    
    Returns:
        bool: 조건을 하나라도 만족하면 True, 모두 만족하지 않으면 False
    """
    prompt = f"""
    아래 질문이 다음 두 조건 중 하나라도 만족하는지 판단하세요. 하나라도 만족하면 'True', 모두 만족하지 않으면 'False'를 반환하세요.

    조건:
    1. 리포트 또는 보고서 직접 작성을 부탁할 것.
    2. 목적을 특정하지 않은 막연한 분석을 요청할 것.

    질문: "{query}"

    답변(반드시 True 또는 False만 출력):
    """
    try:
        # LLM 호출
        response = llm_mini.invoke(prompt)

        # 결과 처리
        result = response.content.strip()
        return result == "True"  # True 또는 False 문자열을 불리언으로 변환
    except Exception as e:
        print(f"오류 발생: {e}")
        return False
