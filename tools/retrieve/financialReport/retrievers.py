"""
다양한 검색기(Retriever) 구현 모듈
"""
import os
import OpenDartReader
from typing import List, Optional, Dict, Any, Callable, Iterable, Sequence, Tuple
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatClovaX
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
from tools.retrieve.financialReport.utils import format_table
from tools.retrieve.financialReport.prompts import (
    REWRITE_PROMPT,
    FINAL_SUMMARY_PROMPT,
    EXTRACT_PROMPT,
    DEFAULT_TITLE,
)
from langchain_core.callbacks import BaseCallbackHandler
from uuid import UUID
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from typing import Callable
from tools.retrieve.financialReport.korean_nlp import KoreanTextAnalyzer

korean_nlp = KoreanTextAnalyzer()

class CustomLLMChainExtractor(LLMChainExtractor):
    """메타데이터의 테이블 정보도 고려하는 문서 압축기"""
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[str, Document], str]] = None,
    ) -> "CustomLLMChainExtractor":
        """LLM으로부터 초기화"""
        
        _prompt = EXTRACT_PROMPT
        
        def custom_get_input(query: str, doc: Document) -> dict:
            """문서와 테이블 정보를 모두 포함하는 입력 생성"""
            table_info = ""
            if "table" in doc.metadata:
                table_info = format_table(doc.metadata["table"])
            
            return {
                "query": query,
                "page_content": doc.page_content,
                "table_info": table_info
            }
        
        _get_input = custom_get_input if get_input is None else get_input
        
        if _prompt.output_parser is not None:
            parser = _prompt.output_parser
        else:
            parser = StrOutputParser()
            
        llm_chain = _prompt | llm | parser
        return cls(llm_chain=llm_chain, get_input=_get_input)
    
class DocumentContentParser:
    """문서 내용을 파싱하고 처리하는 클래스"""
    
    @staticmethod
    def combine_page_contents(docs: List[Document]) -> str:
        """Document 리스트의 모든 page_content를 결합"""
        return " ".join(doc.page_content.strip().strip("\n").strip() for doc in docs)
    
    @staticmethod
    def combine_with_metadata(docs: List[Document]) -> List[dict]:
        """Document 리스트의 내용과 메타데이터를 결합"""
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs
        ]


class RewriteDebugHandler(BaseCallbackHandler):
    """질문 재구성 과정의 디버깅을 위한 콜백 핸들러"""
    def on_chain_end(
        self, 
        outputs: Any, 
        *, 
        run_id: UUID, 
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        """체인 종료 시 호출"""
        if isinstance(outputs, str):
            print("\n" + "="*50)
            print("📝 질문 재구성 결과")
            print("="*50)
            print(f"재구성된 질문: {outputs}")
            print("="*50 + "\n")

class QueryConstructorDebugHandler(BaseCallbackHandler):
    """구조화된 쿼리 생성 과정의 디버깅을 위한 콜백 핸들러"""
    def on_chain_end(
        self, 
        outputs: Any, 
        *, 
        run_id: UUID, 
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        """체인 종료 시 호출"""
        if hasattr(outputs, "filter") or hasattr(outputs, "query"):
            print("\n" + "="*50)
            print("🔍 구조화된 쿼리 생성 결과")
            print("="*50)
            filter_condition = getattr(outputs, "filter", None)
            query = getattr(outputs, "query", None)
            print(f"필터 조건: {filter_condition}")
            print(f"검색 쿼리: {query}")
            print("="*50 + "\n")

    def on_retriever_start(
        self, 
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> Any:
        """검색 시작 시 호출"""
        print("\n" + "="*50)
        print("🚀 문서 검색 시작")
        print("="*50)
        print(f"검색 쿼리: {query}")
        print("="*50 + "\n")

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        """검색 종료 시 호출"""
        print("\n" + "="*50)
        print(f"✅ 검색 완료 - {len(documents)}개 문서 발견")
        print("="*50)
        for i, doc in enumerate(documents, 1):
            print(f"\n📄 문서 {i}:")
            print(f"내용: {doc.page_content[:150]}...")
            print(f"메타데이터: {doc.metadata}")
        print("="*50 + "\n")

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        """검색 오류 발생 시 호출"""
        print("\n" + "="*50)
        print("❌ 검색 오류 발생")
        print("="*50)
        print(f"오류 내용: {str(error)}")
        print("="*50 + "\n")

class RetrievalManager:
    """검색기 관리 클래스"""
    
    def __init__(self, vectorstore, llm: Optional[BaseLanguageModel] = None):
        self.vectorstore = vectorstore
        self.gpt_4o = ChatOpenAI(
            model="chatgpt-4o-latest",
            temperature=0,
        )
        self.gpt_4_turbo = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
        )
        self.gpt_4o_mini = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
        )
        self.gpt_35_turbo = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0,
        )
        self.clova_x = ChatClovaX(
            model="HCX-003",
            temperature=0.1,
            include_ai_filters=False,
        )
        self.korean_nlp = korean_nlp
        self.rewrite_callbacks = [RewriteDebugHandler()]
        self.query_constructor_callbacks = [QueryConstructorDebugHandler()]
        self.dart = OpenDartReader("4925a6e6e69d8f9138f4d9814f56f371b2b2079a")

    def _create_parse_runnable(self):
        """파싱을 위한 Runnable 생성"""
        return RunnableLambda(
            lambda x: x.strip()  # 앞뒤 공백 제거
                .strip('"')      # 따옴표 제거
                .strip('\\')     
                .strip('\\n')    # 개행문자 제거
                .rstrip('*')     # 끝의 별표 제거
                .strip()         # 최종 공백 제거
        ).with_config({"name": "ParseChain", "callbacks": self.rewrite_callbacks})
    
    def _create_rewrite_chain(self):
        """쿼리 재작성 체인 생성"""
        self._parse = self._create_parse_runnable()
        chain = (REWRITE_PROMPT | self.gpt_4o_mini | StrOutputParser() | self._parse)
        return chain
    
    def rewrite_query(self, query: str, title: str = "") -> str:
        """쿼리 재작성 수행"""
        if not title:
            title = DEFAULT_TITLE
        rewriter = self._create_rewrite_chain()
        rewrite_query = rewriter.invoke({"query": query, "title": title})
        return rewrite_query
    
    def create_compression_retriever(self, base_retriever) -> ContextualCompressionRetriever:
        """압축 검색기 생성"""
        compressor = CustomLLMChainExtractor.from_llm(
            self.clova_x,
            )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    
    def create_retriever(self, k: int = 4, use_mmr: bool = True, metadata_filter: Optional[Dict[str, Any]] = None) -> Any:
        """메타데이터 필터를 사용하는 검색기 생성
        
        Args:
            k (int): 반환할 문서 수
            use_mmr (bool): MMR 사용 여부
            metadata (Optional[Dict]): 메타데이터 필터 (예: {"companyName": "삼성전자", "year": 2023})
        """
        # 메타데이터를 ChromaDB 필터 형식으로 변환
        filter_dict = {}
        if metadata_filter:
            filter_conditions = []
            if 'companyName' in metadata_filter:
                filter_conditions.append({"companyName": metadata_filter['companyName']})
            if 'year' in metadata_filter:
                filter_conditions.append({"year": metadata_filter['year']})
            if filter_conditions:
                filter_dict = {"$and": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]
        
        search_kwargs = {
            "k": k,
            "filter": filter_dict,
            "fetch_k": k * 4 if use_mmr else k,
            "lambda_mult": 0.65 if use_mmr else None
        }
        
        if use_mmr:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs
            )
        else:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
                
        return retriever

    def get_retriever_results(self, query: str, k: int = 4, rewrite: bool = True, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """검색기 결과 반환"""
        from tools.retrieve.financialReport.prompts import output_parser
        rcept_no_title = ""
        rcept_no = self.dart.finstate(metadata['companyName'], metadata['year'], reprt_code="11011")
        if not rcept_no.empty:
            rcept_no = rcept_no['rcept_no'].unique()[0]
            rcept_no_title = '\n'.join(self.dart.sub_docs(rcept_no)['title'].tolist())

        if rewrite:
            query = self.rewrite_query(query, rcept_no_title)
        
        final_summary_chain = (
            FINAL_SUMMARY_PROMPT 
            | self.gpt_4o
            | output_parser
        )

        def perform_search(metadata_filter):
            base_retriever = self.create_retriever(
                k=k, 
                metadata_filter=metadata_filter
            )
            compression_retriever = self.create_compression_retriever(base_retriever)
            return compression_retriever.invoke(query)

        # 첫 번째 검색 수행
        results_docs = perform_search(metadata)
        
        # 2024년 검색 결과가 없고, 현재 연도가 2024년인 경우 2023년 데이터로 재검색
        if (not results_docs and 
            metadata and 
            metadata.get('year') == 2024 or metadata.get('year') == 2025):
            metadata_2023 = metadata.copy()
            metadata_2023['year'] = 2023
            results_docs = perform_search(metadata_2023)
            if results_docs:
                # 2023년 데이터를 찾았다는 메시지 추가
                prefix_message = f"[{metadata.get('year')}년 데이터가 없어 {metadata.get('year')-1}년 데이터를 검색했습니다]\n"
        else:
            prefix_message = ""
        
        # 도큐먼트 메타데이터 기반으로 key_information 구성
        report_results = {
            "output": "",
            "key_information": []
        }
        
        # 검색된 문서들의 내용을 결합
        combined_content = []
        
        for doc in results_docs:
            referenced_content = doc.page_content
            if referenced_content and referenced_content.strip():
                metadata = doc.metadata
                report_results["key_information"].append({
                    "tool": "사업보고서",
                    "needed_information": metadata.get("needed_information", "문서 정보"),
                    "referenced_content": referenced_content,
                    "page_number": metadata.get("page_number", "N/A"),
                    "filename": metadata.get("filename", "unknown"),
                    "link": metadata.get("link", "N/A")
                })
                combined_content.append(referenced_content)
        
        # 결합된 내용을 output에 설정
        report_results["output"] = prefix_message + ("\n".join(combined_content) if combined_content else "사업보고서 에이전트에서 관련 정보를 찾을 수 없습니다.")
        
        # 최종 입력 구성
        final_input = {
            "query": query,
            "search_results": report_results,
        }
        
        # 최종 요약 생성
        results = final_summary_chain.invoke(final_input)
        results['output'] = prefix_message + self.korean_nlp.normalize_text(results['output'])
        return results
