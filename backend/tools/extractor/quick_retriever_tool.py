from typing import Any, Dict, List, Optional, Type
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from tools.extractor.financialword_retriever import WebScrapeRetriever
from config.prompts import _quick_retriever_tool_DESCRIPTION
import asyncio

class QuickRetrieverToolInputSchema(BaseModel):
    input_query: str = Field(..., description="검색 문장")
    metadata: Dict[str, Any] = Field(..., description="메타데이터(기업명)")

class QuickRetrieverTool(BaseTool):
    """기업의 재무정보를 빠르게 검색하는 도구"""
    
    name: str = "quick_retriever_tool"
    description: str = _quick_retriever_tool_DESCRIPTION
    args_schema: Type[BaseModel] = QuickRetrieverToolInputSchema
    return_direct: bool = True
    llm: BaseChatModel = Field(default=None, exclude=True)
    financial_terms: List[str] = Field(default=[], exclude=True)
    extract_info_prompt: ChatPromptTemplate = Field(default=None, exclude=True)
    output_parser: Optional[BaseOutputParser] = Field(default=None, exclude=True)
    web_scraper: Optional[WebScrapeRetriever] = Field(default=None, exclude=True)

    # Pydantic에게 "이 모델은 기본 타입 외의 다른 타입들도 허용한다"고 알려주는 역할
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm=llm)

        # self.financial_terms = [
        #     # 1. 안정성비율
        #     "유동비율", "부채비율", "이자보상배율", "자기자본비율", "판관비", "판매비와관리비",
            
        #     # 2. 성장성비율, 수익성비율
        #     "EPS", "영업이익률", "EBITDA", "ROA", "ROE", "ROIC",
        #     "주당순이익", "총자산이익률", "자기자본이익률", "투자자본이익률",
            
        #     # 3. 활동성비율
        #     "총자산회전율"
        # ]

        self.financial_terms = [
            # 0. 돈
            "영업이익", "매출액", "자산총계", "부채총계", "자본총계",
            "자산", "부채", "자본", "매출",

            # 1. 안정성비율
            "유동비율", "부채비율", "이자보상배율", "자기자본비율", "판매비와관리비", "자산총계"
            "판관비",
            
            # 2. 성장성비율, 수익성비율
            "EPS", "영업이익률", "EBITDA", "ROA", "ROE", "ROIC",
            "주당순이익", "총자산이익률", "자기자본이익률", "투자자본이익률",
            
            # 3. 활동성비율
            "총자산회전율"
        ]
        # ROA, ROE, ROIC 같은거는 사람마다 계산이 달라질 수 있으므로, 사업보고서의 정보가 필요하다.
        
        
        self.output_parser = None
        self.web_scraper = WebScrapeRetriever()

    def _extract_info(self, query: str, metadata: Dict[str, Any]):
        """쿼리에서 회사명, 연도, 금융용어를 추출"""
        try:
            # 1. 회사명 추출 (XML 태그 또는 metadata에서)
            company = metadata.get("companyName")
            if not company:
                import re
                company_match = re.search(r'<companyName>(.*?)</companyName>', query)
                if company_match:
                    company = company_match.group(1)
            
            # 2. 최근 5년 연도 추출
            import time
            current_year = time.localtime().tm_year - 2
            years = [current_year - i for i in range(5)]
            print(f"4-5개년: {years}")
            
            # 3. 금융용어 추출 (문자열 매칭)
            financial_terms = set()  # 중복 방지를 위해 set 사용
            for term in self.financial_terms:
                if term in query:
                    if term == "주당순이익":
                        financial_terms.add("EPS")
                    elif term == "총자산이익률":
                        financial_terms.add("ROA")
                    elif term == "자기자본이익률":
                        financial_terms.add("ROE")
                    elif term == "투자자본이익률":
                        financial_terms.add("ROIC")
                    elif term == "판관비":
                        financial_terms.add("판매비와관리비")
                    elif term == "자산":
                        financial_terms.add("자산총계")
                    elif term == "부채":
                        financial_terms.add("부채총계")
                    elif term == "자본":
                        financial_terms.add("자본총계")
                    elif term == "매출":
                        financial_terms.add("매출액")
                    else:
                        financial_terms.add(term)
            
            if all([company, years]) and financial_terms:  # financial_terms가 비어있지 않은지 확인
                return [company, years, list(financial_terms)]  # set을 list로 변환하여 반환
            return None
            
        except Exception as e:
            print(f"정보 추출 중 오류 발생: {str(e)}")
            return None

    async def process_query(self, input_query: str, metadata: Dict[str, Any]):
        """사용자 쿼리를 처리하고 결과를 반환하는 메인 함수"""
        input_data = {"input_query": input_query, "metadata": metadata}
        # 1. 쿼리에서 정보 추출
        extracted_info = self._extract_info(input_query, metadata)
        if not extracted_info or len(extracted_info) < 3:
            return input_data
            
        company, years, financial_terms = extracted_info
        all_results = []  # 루프 밖으로 이동

        for financial_term in financial_terms:
            # 2. 금융용어가 허용된 리스트에 있는지 확인
            if financial_term not in self.financial_terms:
                continue
    
            for year in years:
                year = int(year)
                print(f"추출된 정보: company={company}, year={year}, financial_term={financial_term}")
                
                # 3. WebScrapeRetriever를 통해 데이터 검색
                try:
                    result, url = await self.web_scraper.run(company, financial_term, int(year))
                    if result is None:
                        continue
                    print("result:", result)

                    data = {
                        "query": input_data['input_query'],
                        "company": input_data['metadata']['companyName'],
                        "financial_term": financial_term,
                        "year": year,
                        "result": result,
                        "link": url
                    }
                    all_results.append(data)

                except Exception as e:
                    continue
        
        return all_results if all_results else input_data

    def _run(self, input_query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        동기 실행 메서드 - asyncio.run을 사용하여 비동기 메서드를 실행합니다.
        
        Args:
            input_query (str): 검색 문장
            metadata (Dict[str, Any]): 메타데이터(기업명 등)
        
        Returns:
            Dict[str, Any]: 검색 결과와 관련 정보를 포함하는 딕셔너리
        """
        input_data = {"input_query": input_query, "metadata": metadata}
        try:
            return asyncio.run(self._arun(input_query=input_query, metadata=metadata))
        except Exception as e:
            return input_data

    async def _arun(self, input_query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        비동기 실행 메서드 - 실제 검색과 데이터 처리를 수행합니다.
        
        Args:
            input_query (str): 검색 문장
            metadata (Dict[str, Any]): 메타데이터(기업명 등)
        
        Returns:
            Dict[str, Any]: 검색 결과와 관련 정보를 포함하는 딕셔너리
        """
        input_data = {"input_query": input_query, "metadata": metadata}
        try:
            # 입력 유효성 검사
            if not input_query or not isinstance(input_query, str):
                raise ValueError("유효하지 않은 쿼리입니다.")
            if not metadata or not isinstance(metadata, dict):
                raise ValueError("유효하지 않은 메타데이터입니다.")
                
            # 쿼리 처리 및 결과 반환
            results = await self.process_query(
                input_query=input_query,
                metadata=metadata
            )
            print("self.process_query 결과:", results)
            # 결과 처리 로직
            if results == input_data:
                return input_data
                
            # 여러 결과를 하나의 문자열로 결합
            formatted_contents = []
            # key_information = []
            
            for result in results:
                
                formatted_contents.append(
                    f"{result['company']}의 {result['year']}년 {result['financial_term']}은(는) "
                    f"{result['result']}입니다."
                ) 

            formatted_contents = " ".join(formatted_contents)
            formatted_contents = formatted_contents + "이 데이터의 단위는 (단위 : '%' 또는 '억원')입니다. 적합한 단위를 선택해주세요."
            # 여기바꿔보기
            
            key_information = [
                {
                    "tool": "FnGuide 검색 도구",
                    'source': 'FnGuide',
                    "company": results[0]['company'],
                    "referenced_content": "comp.fnguide.com",
                    "link": results[0]['link']
                }
            ]

            final =  {
                "output": formatted_contents,
                "key_information": key_information
            }

            return final

        except ValueError as ve:
            return input_data
        except Exception as e:
            return input_data