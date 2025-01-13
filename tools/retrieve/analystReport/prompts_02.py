"""
애널리스트 리포트 RAG 시스템의 프롬프트 및 메타데이터 정의
"""

from langchain.chains.query_constructor.schema import AttributeInfo

# 문서 컨텐츠 설명
REPORT_DOCUMENT_DESCRIPTION = """
증권사 애널리스트 리포트로, 다음과 같은 내용을 포함:
- 실적 분석 및 전망 (매출액, 영업이익, EBITDA, 순이익 등)
- 주요 사업부문별 분석 (기초소재, 고부가소재, 생명과학 등)
- 투자의견 및 목표주가
- 산업 동향 및 시장 분석
- 밸류에이션 (PER, PBR, EV/EBITDA 등)
- 리스크 요인
"""

# 메타데이터 필드 정보
METADATA_FIELD_INFO = [
    AttributeInfo(
        name="companyName",
        description="""기업명 검색 조건:
        - 정확한 기업명: eq("companyName", "롯데케미칼")
        - 산업 분류: contain("sector", "화학")
        - 기업 목록: in("companyName", ["롯데케미칼", "SK스퀘어"])
        * 대소문자 구분 없음""",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="리포트 작성 연도",
        type="integer"
    ),
    AttributeInfo(
        name="month",
        description="리포트 작성 월",
        type="integer"
    ),
    AttributeInfo(
        name="sector",
        description="""회사의 산업 분야:
        - 정확한 분야: eq("sector", "반도체")
        - 분야 목록: in("sector", ["반도체", "자동차"])""",
        type="string"
    ),
    AttributeInfo(
        name="element_type", 
        description="""문서 요소 유형:
        - text: 텍스트 내용
        - image: 그래프/차트""",
        type="string"
    ),
    AttributeInfo(
        name="stockbroker",
        description="""증권사 검색 조건:
        - 정확한 증권사명: eq("stockbroker", "유안타증권")
        - 증권사 목록: in("stockbroker", ["유안타증권", "키움증권"])""",
        type="string"
    )
]

# 검색 예제
SEARCH_EXAMPLES = [
    (
        "롯데케미칼의 3분기 실적 전망은?",
        {
            "query": "3분기 매출액 영업이익 실적 전망",
            "filter": 'and(eq("companyName", "롯데케미칼"), eq("year", 2024), gte("month", 7), lte("month", 9))',
            "limit": 3
        }
    ),
    (
        "화학업종 최근 투자의견 변경 내역 보여줘",
        {
            "query": "투자의견 목표주가 변경",
            "filter": 'and(eq("sector", "화학/정유"), eq("element_type", "text"))',
            "limit": 5
        }
    ),
    (
        "현재 NCC 마진 추이가 어떻게 되나요?",
        {
            "query": "NCC 마진 스프레드 추이",
            "filter": 'eq("element_type", "image")',
            "limit": 3
        }
    ),
    (
        "2022년부터 2024년까지 우리금융지주의 배당수익률과 DPS 추이 보여줘",
        {
            "query": "우리금융지주 배당수익률 DPS 추이",
            "filter": 'and(eq("companyName", "우리금융지주"), eq("year", 2024), eq("element_type", "image"))',
            "limit": 1
        }
    ),
    (
        "우리금융지주 PBR 밴드 차트와 증권사별 의견은?",
        {
            "query": "PBR 밴드 차트 증권사 투자의견",
            "filter": 'and(eq("companyName", "우리금융지주"), eq("year", 2024))',
            "limit": 3
        }
    )
]