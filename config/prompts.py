# Tool Descriptions 
_Financial_RAG_DESCRIPTION = """
 - financial_report_search(query: str, company: str, year: int, key_word: str) -> str
 - 사업보고서, 재무보고서를 검색하는 검색기입니다.
 - 재무제표에서 검색하지 못한 정보, 각 기업별로 가지고 있는 고유한 정보들이 있습니다.
 - 쿼리는 원하는 하나의 정보만 입력해야 합니다.
 - key_word는 정확히 검색하고자하는 키워드를 입력해주세요
"""

_Financial_TABLE_DESCRIPTION =  """
- 재무제표 검색을 위한 질문을 해주세요
financial_statement_search(query: str, company: str, year: int) -> str

- 검색 시 주의사항:
  * 연도 검색 시 최신 연도(2023년)부터 검색하면 이전 3년치 데이터(2021-2023)가 함께 조회되어 더 정확한 결과를 얻을 수 있습니다
  * 2021년 이전 데이터 검색 시에도 2023년부터 조회하여 비교 분석하세요
"""

_COMBINED_FINANCIAL_REPORT_DESCRIPTION = """
combined_financial_report_search(query: str, company: str, year: int) -> str
- 사업보고서와 재무제표를 통합 검색하는 도구입니다.
- 입력된 쿼리에서 괄호로 분활된 금융 용어와 상위 금융 용어를 적절하게 검색할 수 있도록 쿼리를 재작성해야합니다.
- 지정된 연도를 기준으로 이전 2개년 데이터도 자동으로 함께 검색됩니다. (예: 2023년 지정 시 2021,2022,2023년 데이터 검색)
- 예시1:
  combined_financial_report_search(query: "2021, 2022, 2023년 연구개발비", company: "명문제약", year: 2023)
- 예시2:
  사용자 질문 : 한화의 총자산과 부채비율((총부채/자기자본) x 100)의 변동 패턴을 알려주세요
  combined_financial_report_search(query: "2021, 2022, 2023년 데이터를 총자산, 총부채, 자기자본, 부채비율을 찾아주세요", company: "한화", year: 2023)
- 예시 3:
  사용자 질문 : 카카오는 실제 빚이 얼마나 되나요? 현금성자산을 제외하고 알려주세요.
  combined_financial_report_search(query: "2023년 부채총계, 현금성자산을 찾아주세요", company: "카카오", year: 2023)
- 예시 4:
  사용자 질문 : 롯데손해보험의 당기순이익을 기준으로 ROA((당기순이익 / 총자산) × 100)를 구해주세요
  combined_financial_report_search(query: "2023년 당기순이익, 총자산, ROA를 찾아주세요", company: "롯데손해보험", year: 2023)
- 예시 5:
  사용자 질문 : HL만도의 최근 연결 재무제표에서 주당순이익(당기순이익 / 가중평균유통주식수) (EPS(당기순이익 / 가중평균유통주식수))은 얼마로 표시되어 있나요?
  combined_financial_report_search(query: "2023년 당기순이익, 총자산, 주당순이익, EPS를 찾아주세요", company: "HL만도", year: 2023)
"""

_Analyst_RAG_DESCRIPTION = """
  "report_rag_search(query: str, company: str, year: int) -> str: "
- 시장 동향, 경쟁사 분석, 투자전략, 특히 맥락좌 전망을 이해하는 데 유리합니다. 
- 외부 요인에 대한 영향, 내부요인에 대한 변화 분석을 합니다. 
- 애널리스트 리포트에서 관련 정보를 검색합니다:
  * 텍스트 정보: 기업의 실적 전망, 목표주가 및 투자의견, 산업 전망, 밸류에이션 전망 등 미래 예측 정보
  * 그래프/차트 정보: 매출/영업이익 추이, 주가 차트, PER/PBR 밴드, 실적 추정치 변경 등 시각적 데이터
  
- 증권사 애널리스트들의 다음과 같은 전문적 분석과 의견을 검색합니다:
  * 정량적 분석: 재무제표 분석, 실적 전망치, 밸류에이션 수준
  * 정성적 분석: 산업 동향, 경쟁 현황, 리스크 요인
  * 투자포인트: 목표주가 산정 근거, 주가 촉매제, 투자 위험 요인
  
- 시각적 데이터는 다음과 같은 인사이트를 제공합니다:
  * 시계열 추세: 매출, 이익, 마진율 등의 장기 트렌드
  * 상대 비교: 업종 평균 대비 밸류에이션 프리미엄/디스카운트
  * 주가 흐름: 주가 모멘텀, 거래량, 수급 동향
  
- 검색 결과는 관련도순으로 정렬되며, 텍스트와 그래프 정보를 통합적으로 분석하여 제시합니다.
ex. 삼성전자 실적 전망
"""

_fintool_DESCRIPTION = """
    "fin_tool(query: str, config: Optional[dict] = None) -> str:\n"
    "- 사용자의 질문(query)에서 금융 용어를 찾아 치환한 후, 그 결과를 반환합니다.\n"
    "- Parameters:\n"
    "  * query: str\n"
    "  * config: Optional[dict]\n"
    "- Returns:\n"
    "  * str: 치환된 결과 문자열 (LLM이 추론한 change)\n"
"""

_query_processor_tool_DESCRIPTION = """
    query_processor_tool(query: str) -> str:
    - 쿼리에서 회사명, 연도 등의 정보를 추출하여, 원본 쿼리에 태그를 달아 변환합니다.
"""

_math_tool_DESCRIPTION = """
    calculator_graph_tool(query: str, context: Optional[list] = None) -> str:
    이 도구는 복잡한 계산과 금융 데이터 조회를 수행할 수 있습니다.
    수학적 계산, 재무제표 분석, 데이터 검색 등을 수행할 수 있습니다.
    입력은 자연어 질문 형태로 제공하면 됩니다."""

_WEB_SEARCH_TOOL_DESCRIPTION = (
    "web_search(query: str, company: str ) -> str:\n"
    "    - 웹에서 관련 정보를 검색하여, 검색 결과를 반환합니다.\n"
    "    - 진출 시장, 동향, 분위기, 이슈 등 사회적 이슈가 질문에 포함되어 있으면 웹 검색을 통해 정보를 추출해주세요\n"
) 

_ANALYZER_DESCRIPTION = """
 - CombinedStockAnalyzer(query: str) -> str:
 - 주어진 기업의 주가를 분석하고 개선사항을 제안합니다.
 - 주가가 왜 올랐고, 떨어졌는지 분석합니다. 
 - 기업의 경쟁, 동일 업종 기업의 주가 비교 분석을 통해 주가 움직임을 분석합니다.
 - 가치평가 기법을 기반으로 가치를 분석하고, 주가를 예측합니다. 
 - 주가 분석 결과는 문자열 형태로 반환됩니다.
"""

_PLANNING_CANDIDATES_PROMPT = """
You are a task planning assistant. Your role is to break down user queries into specific, sequential tasks.
찾는 기간은 두 타입이 있습니다. 1. 연도 2. 사업보고서 회계 기준 ( ~기 ex. 57기 56기 ) 같은 기간 별로 비교하세요 

Rules:
1. Each task should be clearly numbered and start with "1"
2. Tasks should be simple and focused on one specific action
3. Avoid redundant tool usage
4. 최소 3개의 계획 후보군을 생성해야 합니다. 


Examples:
query: 제주은행과 기업은행의 매출을 비교해줘
- task1: 제주은행의 매출 추출
- task2: 기업은행의 매출 추출
- task3: 두 매출 비교

query: 제주은행의 전년대비 매출 증가율을 구해봐
- task1: 제주은행의 올해 매출 추출
- task2: 제주은행의 전년 매출 추출
- task3: 올해 매출 / 전년 매출 *100 (증가율 계산)

query: 제주은행의 56기 매출을 전기와 비교해줘 
- task1 : 제주은행 56기 매출 
- task2 : 제주은행 55기 매출 
- task3 : 56기 매출 / 55기 매출 *100 (증가율 계산)

query: 삼성전자 사업영역을 알려줘
- task1: 삼성전자 사업영역 추출

query: 삼성전자 ROE를 알려줘
- task1: 삼성전자 당기 순이익 추출
- task2: 삼성전자 자기자본 추출
- task3: 삼성전자 ROE 계산

Please analyze the given query and break it down into appropriate tasks following this format:
- task1: [specific action]
- task2: [specific action]
...
"""

MARKET_Data_Tool_PROMPT = """
    market_data_tool(query: str) -> str
    금융 시장 데이터를 분석하고 사용자 질의에 답변합니다.
        주요 기능:
    1. 주식 데이터 분석
       - 개별 종목의 주가 추이 분석
       - 주가 변동률 계산
       - 거래량/거래대금 분석
       - 시가총액 정보 제공
    
    2. 환율 데이터 분석
       - 달러/원화 환율 추이 분석
       - 환율 변동성 분석
       - 기간별 환율 비교
    
    최적화된 쿼리 유형:
    1. 주식 관련:
       - "기업명의 최근 N개월/년간 주가 추이 분석"
       - "기업명의 52주 최고/최저가"
       - "기업명의 거래량 추이"
       - "기업명의 시가총액 변화"
    
    2. 환율 관련:
       - "최근 N개월/년간 환율 동향"
       - "환율의 최고점/최저점"
       - "환율의 변동성 분석"
       
  market_data_tool 주의사항:
    - 주식 데이터는 한국 시장 기준
    - 환율은 달러/원화 기준
    - 복잡한 수학적 계산이 필요한 경우 math_tool과 함께 사용 권장
"""

_SameSectorComplare_ANL_DESCRIPTION = """
SameSectorAnalyzer(self, query: str, company:str, year:int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
  :  기업의 동일 업종 내 위치와 특징, 경쟁사와의 비교, 시장 내 차별성을 파악하려는 질문이 들어왔을 때 이 Tool을 사용합니다.
    - 기능 
      경쟁사와의 영업이익률, 매출액, 영업이익, 매출액 증가율, 시가총액 비교 데이터 
    
"""

_STOCK_ANL_DESCRIPTION = """
StockAnalyzerTool(self, query: str, company:str, year:int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
 - 주가 변동 원인 분석에 대한 질문이 들어왔을 때 이 Tool을 사용합니다.
 - 기능 
   최근 30일간 주가 변동이 큰 날짜를 선택한 후, 주가 변동 원인을 분석합니다.
"""

_CombinedAnalysisTool_DESCRIPTION = """
CombinedAnalysisTool(self, query: str, company:str, year:int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str: "
- 이 툴은 특정 기업의 재무 상태와 가치를 종합적으로 분석하기 위한 통합 도구입니다
- 기능 
  기업 가치 평가 ( 1. 회귀분석 기반 PER,PBR 예측 기반, 2. DCF 기반, 3. 현재 주가 수준 평가)
"""
