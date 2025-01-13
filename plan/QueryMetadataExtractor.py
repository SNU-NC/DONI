from typing import Dict, Tuple, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
import json
import unicodedata
from datetime import datetime
from rapidfuzz import process, fuzz

class QueryMetadata(BaseModel):
    """쿼리에서 추출된 메타데이터"""
    company_name: Optional[str] = Field(None, description="추출된 회사명")
    year: Optional[str] = Field(None, description="추출된 연도")
    cleaned_query: str = Field(..., description="메타데이터가 제거된 정제된 쿼리")

class CompanyNameMatcher:
    """Plan B: 룰베이스로 회사명 매칭하는 클래스"""
    def __init__(self, company_names_file="data/all_company_names.json"):
        self.companies = self._load_company_names(company_names_file)
        self.kor_to_eng = {
            "에이": "A", "비": "B", "씨": "C", "디": "D", "이": "E", "에프": "F",
            "지": "G", "에이치": "H", "아이": "I", "제이": "J", "케이": "K",
            "엘": "L", "엠": "M", "엔": "N", "오": "O", "피": "P", "큐": "Q",
            "아르": "R", "에스": "S", "티": "T", "유": "U", "브이": "V",
            "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z"
        }
        self.eng_to_kor = {v: k for k, v in self.kor_to_eng.items()}

    def _normalize_str(self, s):
        return unicodedata.normalize('NFC', s)

    def _load_company_names(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        companies = data["companies"]
        return [self._normalize_str(c) for c in companies]

    def _apply_mapping(self, query, mapping):
        result = query
        items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
        for k_str, v_str in items:
            if k_str in result:
                result = result.replace(k_str, v_str)
        return result

    def _generate_candidates(self, query):
        candidates = [query]
        
        # 한글->영문 매핑
        kor_to_eng_query = self._apply_mapping(query, self.kor_to_eng)
        if kor_to_eng_query != query:
            candidates.append(kor_to_eng_query)
        
        # 영문->한글 매핑
        eng_to_kor_query = query
        for eng_char, kor_str in self.eng_to_kor.items():
            if eng_char in eng_to_kor_query:
                eng_to_kor_query = eng_to_kor_query.replace(eng_char, kor_str)
        if eng_to_kor_query != query:
            candidates.append(eng_to_kor_query)
        
        return list(set(candidates))

    def match_company_name(self, query: str) -> Tuple[Optional[str], float]:
        query_norm = self._normalize_str(query)
        candidates = self._generate_candidates(query_norm)
        
        best_overall_match = None
        best_overall_score = -1
        
        for candidate in candidates:
            candidate_norm = self._normalize_str(candidate)
            best_match, score, idx = process.extractOne(
                candidate_norm, 
                self.companies, 
                scorer=fuzz.WRatio
            )
            if score > best_overall_score:
                best_overall_score = score
                best_overall_match = best_match
                
        # 임계값 설정 (예: 80)
        if best_overall_score < 80:
            return None, best_overall_score
            
        return best_overall_match, best_overall_score

class LLMMetadataExtractor:
    """PlanA : LLM을 사용하여 메타데이터 추출"""
    def __init__(self, llm: BaseChatModel, company_names_file="data/all_company_names.json"):
        self.llm = llm.with_structured_output(QueryMetadata)
        self.current_year = str(datetime.now().year)
        
        # 회사명 목록 로드
        with open(company_names_file, "r", encoding="utf-8") as f:
            self.companies = json.load(f)["companies"]
            
        # 프롬프트에 회사 목록과 현재 연도 포함
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 사용자의 쿼리에서 회사명, 연도 정보를 추출하고 이를 제거한 정제된 쿼리를 생성하는 전문가입니다.
            현재 연도는 {current_year}입니다.
            
            아래는 허용된 회사명 목록입니다:
            {companies}
            
            당신의 임무는 다음과 같습니다:

            1. company_name: 회사명을 추출하고 위 목록에서 정확히 일치하는 항목을 찾습니다.
               - 정확히 일치하는 항목이 없다면, 다음 변환을 순차적으로 시도하여 일치하는 항목을 찾습니다:
                 a) 한글 발음 → 영문
                   예: "에스케이" → "SK" 변환 후 확인
                   예: "엔에이치엔" → "NHN" 변환 후 확인
                   예: "엘브이엠씨" = "엘(L) + 브이(V) + 엠(M) + 씨(C)" → "LVMC" 변환 후 확인
                 b) 영문 → 한글 발음
                   예: "SK" → "에스케이" 변환 후 확인
                 c) 회사 약칭/구명칭 → 정식명칭
                   예: "삼전" → "삼성전자" 확인
                   예: "하이닉" → "SK하이닉스" 확인
                   예: "포스코" → "POSCO홀딩스" 확인 
                   예: "현대차" → "현대자동차" 확인
               - 모든 변환 시도 후에도 목록에서 일치하는 항목을 찾지 못하면 None을 반환
               - 복수의 회사가 언급되면 각각에 대해 위 과정을 수행하여 리스트로 반환
             
            2. cleaned_query: 메타데이터가 제거된 정제된 쿼리를 생성합니다.
               - 회사명과 연도를 완전히 제거
               - "~의", "~에 대한" 등의 조사도 함께 제거

            입력: {query}

            응답 시 반드시 지정된 회사명 목록에서 찾은 정확한 회사명만 반환하세요.
            """),
        ])

        
        # 룰 베이스 매처는 백업용으로 유지
        self.company_matcher = CompanyNameMatcher(company_names_file)

    def extract_metadata(self, query: str) -> Tuple[str, Dict[str, str]]:
        print("\n=== 메타데이터 추출 시작 ===")
        print(f"입력 쿼리: {query}")
        
        current_year = str(datetime.now().year)
        prev_year = str(int(current_year) - 1)
        next_year = str(int(current_year) + 1)
        
        # LLM을 사용하여 메타데이터 추출
        result = self.llm.invoke(
            self.prompt.format(
                query=query,
                companies=", ".join(self.companies),
                current_year=current_year,
                prev_year=prev_year,
                next_year=next_year
            )
        )
        
        print("\n[LLM 추출 결과]")
        print(f"- 회사명: {result.company_name}")
        print(f"- 연도: {result.year}")
        print(f"- 정제된 쿼리: {result.cleaned_query}")
        
        # 메타데이터 딕셔너리 생성
        metadata = {}

        # company_name이 None인 경우 룰 베이스로 재시도
        # if result.company_name is None:
        #     print("\n[B. 룰 베이스 매칭 시도]")
        #     company_match, score = self.company_matcher.match_company_name(query)
        #     if company_match is not None:
        #         print(f"- 매칭된 회사명: {company_match}")
        #         print(f"- 매칭 점수: {score}")
        #         metadata["companyName"] = company_match
        #         extraction_method = "Rule-based"
        #     else:
        #         print("- 회사명 매칭 실패")
        # else:
        #     metadata["companyName"] = result.company_name
        
        # 회사명이 실제 목록에 있는지 최종 확인
        if result.company_name:
            if isinstance(result.company_name, list):
                valid_companies = [comp for comp in result.company_name if comp in self.companies]
                if valid_companies:
                    metadata["companyName"] = valid_companies
            else:
                if result.company_name in self.companies:
                    metadata["companyName"] = result.company_name
            
        # 연도 처리
        if result.year:
            metadata["year"] = result.year
            
        # "올해" 관련 키워드가 있는지 확인하고 현재 연도 추가
        year_keywords = ["올해", "이번년도", "금년", "올 해"]
        if any(keyword in query for keyword in year_keywords):
            metadata["year"] = current_year
        
        print("\n[최종 결과]")
        print(f"- 메타데이터: {metadata}")
        print(f"- 정제된 쿼리: {result.cleaned_query}")
        print("=== 메타데이터 추출 완료 ===\n")
            
        return result.cleaned_query, metadata