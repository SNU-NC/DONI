from typing import Optional, List, Dict, Type, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
import os
import requests
from abc import ABC, abstractmethod
from config.prompts import _WEB_SEARCH_TOOL_DESCRIPTION

load_dotenv()

class WebSearchInputSchema(BaseModel):
    query: str = Field(..., description="구글, 네이버 검색에 최적화된 문장")
    company: str = Field(..., description="검색할 회사명")

class SearchEngine(ABC):
    """Abstract base class for search engines"""
    
    @abstractmethod
    def search(self, query: str) -> str:
        """Execute search and return results"""
        pass

class GoogleSearch(SearchEngine):
    """Google search implementation"""
    
    def __init__(self, api_key: str, cx: str):
        self.api_key = api_key
        self.cx = cx
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str) -> List[dict]:
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cx,
            "num": 5
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            results = response.json()
            
            if "items" not in results:
                return []
                
            search_results = []
            for item in results["items"][:5]:
                link = item.get('link', 'No link')
                # 블로그 도메인 필터링
                if 'tistory.com' in link or 'blog.naver.com' in link:
                    continue
                    
                search_results.append({
                    "tool": "검색 도구[구글]",
                    "title": item.get('title', 'No title'),
                    "link": link,
                    "referenced_content": item.get('snippet', 'No description')
                })
                
            return search_results
            
        except requests.exceptions.RequestException as e:
            return []

class NaverSearch(SearchEngine):
    """Naver search implementation"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/webkr.json"
    
    def search(self, query: str) -> List[dict]:
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        
        params = {
            "query": query,
            "display": 5
        }
        
        try:
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            results = response.json()
            
            if "items" not in results or not results["items"]:
                return []
                
            search_results = []
            for item in results["items"][:5]:
                link = item.get('link', 'No link')
                # 블로그 도메인 필터링
                if 'tistory.com' in link or 'blog.naver.com' in link:
                    continue
                    
                title = item.get("title", "No title").replace("<b>", "").replace("</b>", "")
                description = item.get("description", "No description").replace("<b>", "").replace("</b>", "")
                
                search_results.append({
                    "tool": "검색 도구[네이버]",
                    "title": title,
                    "link": link,
                    "referenced_content": description
                })
                
            return search_results
            
        except requests.exceptions.RequestException as e:
            return []


class WebSearchTools(BaseTool):
    """Main class for managing search tools"""
    
    name: str = "web_search"
    description: str = _WEB_SEARCH_TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = WebSearchInputSchema
    return_direct: bool = True
    logger: logging.Logger = Field(default=None, exclude=True)

    # Pydantic 필드로 추가
    google_api_key: str = Field(default='')
    google_cx: str = Field(default='')
    naver_client_id: str = Field(default='')
    naver_client_secret: str = Field(default='')
    llm: BaseChatModel = Field(default=None)
    google_engine: Optional[GoogleSearch] = Field(default=None)
    naver_engine: Optional[NaverSearch] = Field(default=None)

    def __init__(self, llm: BaseChatModel):
        super().__init__()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CSE_ID")
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        # Validate environment variables
        if not all([
            self.google_api_key,
            self.google_cx,
            self.naver_client_id,
            self.naver_client_secret
        ]):
            raise ValueError("Missing required environment variables")
        
        # Initialize search engines
        self.google_engine = GoogleSearch(self.google_api_key, self.google_cx)
        self.naver_engine = NaverSearch(self.naver_client_id, self.naver_client_secret)
        self.llm = llm

    def _create_search_function(self, engine: SearchEngine):
        """Create a search function for the given engine"""
        def search(
            query: str,
            context: Optional[List[str]] = None
        ) -> str:
            if context:
                query = f"{query} {' '.join(context)}"
            return engine.search(query)
        return search
    
    def get_tools(self) -> List[StructuredTool]:
        """Get list of search tools"""
        
        # Create Google search tool
        google_tool = StructuredTool.from_function(
            name="google_search",
            func=self._create_search_function(self.google_engine),
            description="Search Google for web results. Useful for finding general information and English content.",
            args_schema=dict(
                query=str,
                context=Optional[List[str]]
            )
        )
        
        # Create Naver search tool
        naver_tool = StructuredTool.from_function(
            name="naver_search",
            func=self._create_search_function(self.naver_engine),
            description="Search Naver for web results. Useful for finding Korean content and local information.",
            args_schema=dict(
                query=str,
                context=Optional[List[str]]
            )
        )
        
        return [google_tool, naver_tool]
    
    def _format_results(self, google_results: List[dict], naver_results: List[dict]) -> dict:
        """검색 결과를 구조화된 딕셔너리 형태로 포맷팅"""
        search_results = {
            "output": "",
            "key_information": []
        }
        print("google_results:" , google_results, "naver_results:" , naver_results)
        # 모든 검색 결과 합치기
        search_results["key_information"] = google_results + naver_results
        
        # 전체 출력 텍스트 생성
        all_contents = []
        for info in search_results["key_information"]:
            all_contents.append(f"[{info['title']}] {info['referenced_content']}")
        
        search_results["output"] = "\n\n".join(all_contents) if all_contents else "검색 결과를 찾을 수 없습니다."
        print("search_results:" , search_results)
        return search_results

    def _run(
        self,
        query: str,
        company: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        try:
            search_query = company + " " + query
            
            # 검색 실행
            print(f"검색 쿼리: {search_query}")
            google_results = self.google_engine.search(search_query)
            naver_results = self.naver_engine.search(search_query)
            
            # 결과 조합
            combined_results = self._format_results(google_results, naver_results)
            
            return combined_results
            
        except Exception as e:
            error_msg = f"검색 에이전트 오류 발생: {str(e)}\n상세 오류: {type(e).__name__}"
            if self.logger:  # logger가 None이 아닐 때만 로깅
                self.logger.error(error_msg)
            raise ToolException(error_msg)
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """비동기 실행은 동기 실행과 동일한 로직 사용"""
        return self._run(query=query, run_manager=run_manager)