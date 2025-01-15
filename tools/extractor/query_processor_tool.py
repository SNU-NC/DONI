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
    4. 쿼리에 나와있는 형태 그대로 반환해야합니다.
     
    출력 형식:
    ["회사명1", "회사명2"] 형태로만 정확히 반환

    예시:
    입력: 삼성전자 주가가 어떻게 되나요?
    출력: ["삼성전자"]
    
    입력: 카카오는 실제 빚이 얼마나 되나요? 현금성자산을 제외하고 알려주세요.
    출력: ["카카오"]
     
    입력: 엥스케이하이닉스 주가 어떤가요?
    출력: ["엥스케이하이닉스"]

    입력: LG전자와 SK하이닉스 중 어느 회사가 더 좋을까요?
    출력: ["LG전자", "SK하이닉스"]

    입력: 엘지와 삼성의 영업이익률 추이가 궁금합니다
    출력: ["엘지", "삼성"]
     
    입력: 카카오 종속기업중에 카카오모빌리티의 진출 시장과 그 회사의 총자산에 대해서 알려줘
    출력: ["카카오", "카카오모빌리티"]

    """),
])
        # 2) HyperCLOVA-X용 후보 선택 프롬프트
        self.candidate_selection_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 주어진 기업명과 가장 일치하거나 유사한 회사를 후보 목록에서 찾는 전문가입니다.

주어진 후보 목록에서 다음 우선순위로 가장 적절한 회사를 하나만 선택하십시오:
1순위: 입력된 기업명과 목록의 회사명이 정확히 일치하는 경우
- 다음은 모두 동일한 회사로 간주하고 목록에 있는 형태를 반환합니다:
  * "네이버", "NAVER", "naver", "Naver" → 목록에 있는 "NAVER" 반환
  * "삼성", "SAMSUNG", "samsung" → 목록에 있는 형태로 반환
- 띄어쓰기는 무시합니다
2순위: 정확히 일치하는 것이 없는 경우, 입력된 기업명과 가장 유사한 회사

후보 회사 목록:
{candidates}

입력된 기업명: {query}

### 응답 규칙 ###
1. 후보 목록에 회사가 하나만 있다면 → 무조건 그 회사를 반환
2. 후보 목록에 여러 회사가 있다면:
   - 정확히 일치하는 회사가 있을 경우 → 그 회사를 반환
   - 정확히 일치하는 회사가 없을 경우 → 가장 유사한 회사를 반환
3. 회사명만 반환하고 다른 텍스트는 절대 포함하지 마십시오.
4. 따옴표나 구두점 없이 회사명만 정확히 반환하십시오.
     
### 출력 형식 ###
회사명
     
### 응답 전 확인사항 ###
- 후보가 1개라면 그 회사를 그대로 반환했는가?
- 회사명 외의 다른 텍스트가 포함되지 않았는가?
- 따옴표나 구두점이 포함되지 않았는가?
    """),
])  

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
                    k=20, 
                )
                if not hybrid_results:
                    print("- 적절한 후보를 찾지 못했습니다.")
                    continue

                candidates = [company for company, _ in hybrid_results]
                print(f"- 추출된 후보 수: {len(candidates)}")
                print(f"- 추출된 후보 목록: {candidates}")
                
                # 후보가 1개라면 Step 3 작업 패스 (그 회사를 그대로 반환)
                if len(candidates) == 1:
                    final_companies.append(candidates[0])
                    print(f"- 최종 선택된 회사명: {final_companies}")
                    continue

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