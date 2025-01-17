import json
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from config.prompts import _fintool_DESCRIPTION


class FinancialQuery(BaseModel):
    """The structure for querying financial terms."""
    change: str = Field(..., description="The updated query with matched financial terms replaced (if any).")
    terms: list[str] = Field(..., description="The matched financial terms from the user's query.")


# 1) JSON 파일 로드
with open("data/금융용어_최신.json", "r", encoding="utf-8") as file:
    financial_data = json.load(file)

# 2) 사전 형태로 변환
financial_dict = {entry["term"]: entry["formula"] for entry in financial_data}

# [중요] 여러 용어를 찾기 위해, 간단히 '문자열 치환'을 활용하는 함수
def embed_financial_formulas_in_query(query: str, fin_dict: dict) -> str:
    """
    query 문장에서 fin_dict 내 존재하는 금융 용어를 찾아,
    각 용어 뒤에 (공식)을 삽입하여 반환한다.
    """
    updated_query = query
    found_term = False

    for term, formula in fin_dict.items():
        # term이 문장에 존재하면 치환
        if term in updated_query:
            # 치환문: ROE -> ROE(ROE = (당기순이익 / 자기자본) × 100) 형태
            updated_query = updated_query.replace(term, f"{term}({formula})")
            found_term = True

    # 치환 결과가 없으면(=found_term==False)이면 원본 그대로
    if not found_term:
        return query
    return updated_query

def get_fin_tool(llm: ChatOpenAI):
    """
    LLM과 결합된 fin_tool을 생성.
    (기본 코드는 그대로 두되, formula 반환 시에 embed_financial_formulas_in_query 사용)
    """

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # 시스템 역할: LLM에게 "사용자의 query 안에 있는 금융 용어"를 찾아서
            # FinancialQuery 형식(JSON)으로 응답하라고 지시.
            f"""당신은 금융 보조원입니다.
아래는 금융 용어 목록입니다:
{', '.join(financial_dict.keys())}

당신의 작업:
1) 사용자 query 안에 이 목록에 나열된 금융 용어가 포함되어 있는지 식별하세요.
2) 용어가 발견되면, 위에 명시된 정규(canonical) 형태로 대체(또는 유지)하여 'change' 문자열을 만드세요.
3) 'change'(수정된 쿼리)와 'terms'(일치된 용어들)을 JSON 스키마에 맞춰 작성하세요.
4) 용어가 전혀 일치하지 않을 경우, 'change'는 원본 query를 그대로 사용하고, 'terms'는 빈 리스트로 두세요.

출력은 다음 JSON 스키마를 반드시 정확히 따라야 합니다:
{{{{
    "change": "<수정된 혹은 원본 query>",
    "terms": ["<일치된_용어1>", "<일치된_용어2>", ...]
}}}}
"""
        ),
        ("user", "{query}"),
    ]
)

    extractor = prompt | llm.with_structured_output(FinancialQuery)

    def retrieve_formula(query: str, config: Optional[dict] = None) -> str:
        # 1) LLM에 term 추출 (단일 용어 기준; 필요시 수정 가능)
        query_changed = extractor.invoke({"query": query}, config)
        extracted_query = query_changed.change

        # 2) 실제론 여러 용어를 처리하기 위해 embed_financial_formulas_in_query 호출
        updated_query = embed_financial_formulas_in_query(extracted_query, financial_dict)

        # 3) 만약 embed 결과가 원본과 동일하다면 => '매칭 용어 없음'
        #    그렇지 않다면 => 치환된 문자열 반환
        if updated_query == query:
            # "금융 용어 관련 식이 없다." 식으로 처리할 수도 있고,
            # 여기서는 원본을 그대로 formula에 담아 반환
            return query  # 매칭된 용어가 없으니 원본 그대로
        else:
            return updated_query  # 치환된 최종 결과 문장

    return StructuredTool.from_function(
        name="fin_tool",
        func=retrieve_formula,
        description=_fintool_DESCRIPTION,
    )