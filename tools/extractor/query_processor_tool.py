from typing import Any, Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
import json
import unicodedata
from datetime import datetime
from tools.retrieve.financialReport.company_name_vector_store import CompanyVectorStore
from config.prompts import _query_processor_tool_DESCRIPTION

class QueryProcessor:
    def __init__(self, llm: BaseChatModel, llm_clova: BaseChatModel):
        self.llm = llm  # GPT-4o-mini
        self.llm_clova = llm_clova # clovaX
        self.company_vector_store = CompanyVectorStore()
        self.company_vector_store.load_or_create()

        # 1) GPT-4o-mini용 기업명 추출 프롬프트
        # 목록은 주지 않고, 예제만 줘서 리스트로 반환하도록
        self.extract_company_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 사용자의 쿼리에서 검색이 필요한 회사명을 추출하는 전문가입니다.
    
    당신의 임무는 다음과 같습니다:
    1. 입력된 쿼리에서 검색이 필요한 회사명을 찾습니다.
    2. 여러 회사가 언급된 경우 질문의 의도를 이해하여 필요한 회사명만 추출합니다.
    3. 반드시 리스트 형태로 반환합니다.
    
    입력 쿼리: {query}

    응답 규칙:
    1. 회사가 하나도 없으면 빈 리스트 []를 반환합니다.
    2. 반드시 리스트 형식으로 응답해야 합니다.
    3. 어떠한 설명이나 추가 텍스트도 포함하지 마십시오.
     
    출력 형식:
    ["회사명1", "회사명2"] 형태로만 정확히 반환

    예시:
    입력: 삼성전자 주가가 어떻게 되나요?
    출력: ["삼성전자"]
    
    입력: 카카오는 실제 빚이 얼마나 되나요? 현금성자산을 제외하고 알려주세요.
    출력: ["카카오"]

    입력: LG전자와 SK하이닉스 중 어느 회사가 더 좋을까요?
    출력: ["LG전자", "SK하이닉스"]

    입력: 엘지와 삼성의 영업이익률 추이가 궁금합니다
    출력: ["LG", "삼성"]

    """),
])
        # 2) HyperCLOVA-X용 후보 선택 프롬프트
        self.candidate_selection_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 주어진 기업명과 가장 일치하거나 유사한 회사를 후보 목록에서 찾는 전문가입니다.

주어진 후보 목록에서 다음 우선순위로 가장 적절한 회사를 하나만 선택하십시오:
1순위: 입력된 기업명과 목록의 회사명이 정확히 일치하는 경우
2순위: 정확히 일치하는 것이 없는 경우, 입력된 기업명과 가장 유사한 회사

후보 회사 목록:
{candidates}

입력된 기업명: {query}

### 응답 규칙
1. 입력된 기업명이 후보 목록에 정확히 일치하는 것이 있다면 반드시 해당 회사를 선택합니다.
   예: 입력된 기업명이 "카카오"이고 목록에 ["카카오", "카카오뱅크"]가 있다면 반드시 "카카오"를 선택
2. 정확히 일치하는 회사가 없을 때만 가장 유사한 회사를 선택합니다.
3. 회사명만 목록에 있는 그대로 정확히 반환해야 합니다.
4. None은 절대 반환하지 않습니다.
5. 어떠한 설명이나 추가 텍스트도 포함하지 마십시오.
6. 따옴표나 기타 구두점 없이 회사명만 정확히 반환하십시오.
7. 출력은 회사명만 반환해야 합니다.
    """),
])  

        # # 메타데이터(연도) 추출 프롬프트
        # self.extract_year_prompt = ChatPromptTemplate.from_messages([
        #     ("system", """
        #             당신은 사용자의 쿼리에서 연도 정보를 추출하는 전문가입니다.
        #             현재 연도는 {current_year}입니다.
                    
        #             당신의 임무는 다음과 같습니다:
                    
        #             - YYYY 또는 YYYY년 형식 인식
        #             - "올해", "이번년도", "금년" 등은 현재 연도({current_year})로 변환
        #             - "작년"은 {prev_year}로 변환
        #             - "내년"도 {prev_year}로 변환
        #             - 연도가 없다면 None을 반환

        #             입력: {query}
        #             응답 시 반드시 다음을 지켜주세요:
        #             1. 정수 형태(integer)로만 반환
        #             """),
        #         ])

    def extract_info(self, query: str) -> Dict[str, str]:
        print("\n=== 기업명 추출 시작 ===")
        print(f"입력 쿼리: {query}")
        
        info = {}
        current_year = str(datetime.now().year)
        prev_year = str(int(current_year) - 1)

        # Step 1: LLM으로 회사명으로 보이는 텍스트 추출
        print("\n[Step 1: LLM으로 회사명 텍스트 추출 중...]")
        company_response = self.llm.invoke(
            self.extract_company_prompt.format(query=query)
        )
        print(f"LLM 응답: {company_response}")
        potential_company_text = company_response.content.strip().strip('"\'')
        
        # 회사 리스트 파싱
        company_list = []
        if potential_company_text.startswith('[') and potential_company_text.endswith(']'):
            company_list = [c.strip().strip('"\'') for c in potential_company_text.strip('[]').split(',')]
        else:
            company_list = [potential_company_text] if potential_company_text else []
        
        print(f"- LLM이 예상한 회사명 목록: {company_list}")

        # 다중 기업 처리를 위한 리스트
        final_companies = []

        # Step 1 결과가 있는 경우
        if company_list not in ([''], []):
            # 각 회사명에 대해 처리
            for potential_company in company_list:
                # Step 2: RuleBased로 쿼리와 가장 유사한 기업명들(상위 20개 후보) 추출
                print(f"\n[Step 2: {potential_company}에 대한 유사 기업명 후보 추출 중...]")
                hybrid_results = self.company_vector_store.hybrid_search(
                    query=potential_company,
                    k=50, 
                )
                if not hybrid_results:
                    print("- 적절한 후보를 찾지 못했습니다.")
                    continue

                candidates = [company for company, _ in hybrid_results]
                print(f"- 추출된 후보 수: {len(candidates)}")
                print(f"- 추출된 후보 목록: {candidates}")
                
                # Step 3: HyperCLOVA-X가 최종 회사명 결정
                print(f"\n[Step 3: {potential_company}에 대한 HyperCLOVA-X 최종 선택 중...]")
                
                clova_response = self.llm_clova.invoke(
                    self.candidate_selection_prompt.format(
                        query=potential_company,
                        candidates=candidates
                    )
                )
                
                print(f"clova_response: {clova_response}")
                final_company = clova_response.content.strip().strip('"\'')
                if final_company and final_company.lower() != 'none' and final_company in candidates:
                    final_companies.append(final_company)
                    print(f"- 최종 선택된 회사명: {final_company}")
                else:
                    print("- 적절한 회사를 찾지 못했습니다.")
        # Step 1 결과가 없는 경우
        else:
            final_companies = []

        # 다중 기업 정보 저장
        if final_companies:
            info["companyNames"] = final_companies

        # 연도 정보 추출 (기존 코드와 동일)
        # year_response = self.llm.invoke(
        #     self.extract_year_prompt.format(
        #         query=query,
        #         current_year=current_year,
        #         prev_year=prev_year
        #     )
        # )
        # extracted_year = year_response.content.strip().strip('"\'')
        # if extracted_year:
        #     try:
        #         extracted_year = int(extracted_year)
        #         info["year"] = extracted_year
        #     except (ValueError, TypeError):
        #         print(f"경고: 연도 값 '{extracted_year}'를 정수로 변환할 수 없어 None으로 설정합니다")
        #         extracted_year = None

        # print("\n[연도 정보 추출 결과]")
        # print(f"- 연도: {extracted_year}")
        
        # # 룰베이스 - 올해 관련 키워드 처리
        # current_year_keywords = ["올해", "이번년도", "금년", "올 해", f"{current_year}년"]
        # if any(keyword in query for keyword in current_year_keywords):
        #     info["year"] = int(current_year)
        
        # # 룰베이스 - 작년 관련 키워드 처리
        # prev_year_keywords = ["작년", f"{prev_year}년"]
        # if any(keyword in query for keyword in prev_year_keywords):
        #     info["year"] = int(prev_year)
            
        return company_list, info
    
    def add_tags(self, query: str, llm_info: List[str], info: Dict[str, str]) -> str:
        """쿼리에 태그 추가"""
        tagged_query = query
        offset = 0  # 태그 추가로 인한 인덱스 오프셋

        # 1. 회사명 태그 추가
        if "companyNames" in info and len(llm_info) == len(info["companyNames"]):
            for original, normalized in zip(llm_info, info["companyNames"]):
                start_idx = query.find(original)
                if start_idx != -1:
                    end_idx = start_idx + len(original)
                    # 태그 추가 (정규화된 회사명으로 대체)
                    tagged_query = (
                        tagged_query[:start_idx + offset] +
                        "<companyName>" +
                        normalized +  # 정규화된 회사명으로 대체
                        "</companyName>" +
                        tagged_query[end_idx + offset:]
                    )
                    offset += len("<companyName>") + len(normalized) + len("</companyName>") - len(original)


        # 2. 연도 태그 추가
        # if "year" in info:
        #     year_str = str(info["year"])
        #     current_year = str(datetime.now().year)
        #     prev_year = str(int(current_year) - 1)
            
        #     # 모든 연도 키워드를 dictionary로 관리
        #     year_keywords = {
        #         "올해": current_year,
        #         "이번년도": current_year,
        #         "금년": current_year,
        #         "올 해": current_year,
        #         "작년": prev_year,
        #         f"{year_str}년": year_str
        #     }
            
        #     # 모든 연도 키워드에 대해 검사
        #     for keyword, year_value in year_keywords.items():
        #         start_idx = query.find(keyword)
        #         if start_idx != -1:
        #             # 태그 추가
        #             tagged_query = (
        #                 tagged_query[:start_idx + offset] +
        #                 "<year>" +
        #                 year_str +  # year_str은 이미 info에서 올바른 값으로 설정됨
        #                 "</year>" + 
        #                 tagged_query[start_idx + len(keyword) + offset:]
        #             )
        #             offset += len("<year>") + len(year_str) + len("</year>") - len(keyword)
        #             break  # 첫 번째로 발견된 키워드만 처리

        return tagged_query
    
    def process_query(self, query: str) ->  Dict[str, Any]:
        """쿼리 처리 로직"""
        # 1) 기업명, 연도 추출
        llm_info, info =self.extract_info(query)
        
        # 2) 태그가 추가된 쿼리 생성
        tagged_query = self.add_tags(query, llm_info, info)

        print("\n[최종 결과]")
        print(f"- 추출된 정보: {info}")
        print(f"- 태그된 쿼리: {tagged_query}")
        print("=== 쿼리 처리 완료 ===\n")
        
        # metadata 구조에 맞게 변환
        metadata = {}
        if "companyNames" in info:
            metadata["companyName"] = info["companyNames"][0]  # 첫 번째 회사만 사용

        print("metadata:", metadata)
           
        # 딕셔너리 형태로 반환
        return {
            "input_query": tagged_query,
            "metadata": metadata
        }

def get_query_processor_tool(llm: BaseChatModel, llm_clova: BaseChatModel) -> StructuredTool:
    processor = QueryProcessor(llm, llm_clova)
    return StructuredTool(
        name="query_processor_tool",
        description=_query_processor_tool_DESCRIPTION,
        func=processor.process_query,
        args_schema=None
    )