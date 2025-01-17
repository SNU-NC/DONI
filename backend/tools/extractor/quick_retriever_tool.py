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

        self.financial_terms = [
            # 1. 안정성비율
            "유동비율", "부채비율", "이자보상배율", "자기자본비율",
            
            # 2. 성장성비율, 수익성비율
            "EPS", "영업이익률", "EBITDA", "ROA", "ROE", "ROIC",
            
            # 3. 활동성비율
            "총자산회전율"
        ]
        
        self.extract_info_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 사용자의 쿼리에서 연도를 추출하는 전문가입니다.
            
            다음 규칙을 따라주세요:
            1. 연도는 숫자 4자리로 추출합니다.
            2. 출력은 연도만 출력합니다.
            3. 추가 텍스트나 특수문자를 포함하지 않습니다.
            
            입력 쿼리: {query}
            
            출력 예시: 2023
            """)
        ])
        
        self.output_parser = None
        self.web_scraper = WebScrapeRetriever()

    def _extract_info(self, query: str, metadata: Dict[str, Any]) -> Optional[List[str]]:
        """쿼리에서 회사명, 연도, 금융용어를 추출"""
        try:
            # 1. 회사명 추출 (XML 태그 또는 metadata에서)
            company = metadata.get("companyName")
            if not company:
                import re
                company_match = re.search(r'<companyName>(.*?)</companyName>', query)
                if company_match:
                    company = company_match.group(1)
            
            # 2. 연도 추출 (LLM)
            chain = self.extract_info_prompt | self.llm
            year_message = chain.invoke({
                "query": query
            })
            year = year_message.content.strip()
            
            # 3. 금융용어 추출 (문자열 매칭)
            financial_term = None
            for term in self.financial_terms:
                if term in query:
                    financial_term = term
                    break
            
            if all([company, year, financial_term]):
                return [company, year, financial_term]
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
            
        company, year, financial_term = extracted_info
        year = int(year)
        print(f"추출된 정보: company={company}, year={year}, financial_term={financial_term}")
        
        # 2. 금융용어가 허용된 리스트에 있는지 확인
        if financial_term not in self.financial_terms:
            return input_data
        
        # 3. WebScrapeRetriever를 통해 데이터 검색
        try:
            result, url = await self.web_scraper.run(company, financial_term, int(year))
            if result is None:
                return input_data
            print("result:", result)
            print("지수투의링크를찾아보자:", url)
            return {
                "query": input_data['input_query'],
                "company": input_data['metadata']['companyName'],
                "financial_term": financial_term,
                "year": year,
                "result": result,
                "link": url
            }
            
        except Exception as e:
            return input_data

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
                
            if "result" in results:
                formatted_content = (f"{results['company']}의 {results['year']}년 {results['financial_term']}은(는) "
                                    f"{results['result']}입니다."
                                    f" 이 정보는 FnGuide에서 확인되었습니다.")
                return {
                    "output": formatted_content,
                    "key_information": [
                        {
                            "tool": "웹검색 재무제표 도구",
                            "company": results['company'],
                            "financial_term": results['financial_term'],
                            "year": results['year'],
                            "link": results['link']
                        }
                    ]
                }
                
            return input_data
                
        except ValueError as ve:
            return input_data
        except Exception as e:
            return input_data