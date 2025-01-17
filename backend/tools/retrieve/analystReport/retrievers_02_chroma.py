"""
리포트 검색기(Retriever) 구현 모듈
"""

from typing import List, Optional, Dict, Any, Sequence, Callable, Union
from uuid import UUID
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers import ContextualCompressionRetriever
from datetime import datetime
import logging


class ReportQueryRewriter:
    """애널리스트 리포트에 특화된 쿼리 재작성기"""
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""당신은 애널리스트 리포트 검색을 위한 전문가입니다.
        주어진 검색어를 애널리스트 리포트에서 자주 사용되는 전문 용어와 표현으로 재구성해주세요.

        예시:
        - 입력: "실적이 어떤가요"
          출력: "매출액 영업이익 순이익 실적 추이 전망"
        - 입력: "투자의견 알려줘"
          출력: "투자의견 목표주가 투자포인트 밸류에이션"
        - 입력: "매출"
          출력: "매출액 영업수익 사업부문별 매출 구성"
        
        원본 검색어: {query}
        
        애널리스트 보고서 용어로 재구성한 검색어:""",
            input_variables=["query"]
        )
        
    def rewrite_query(self, query: str) -> str:
        # 따옴표 제거 및 context 분리
        if ", context=" in query:
            query = query.split(", context=")[0]  # context 부분 제거
        query = query.strip('"')
        
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

class ReportLLMChainExtractor(LLMChainExtractor):
    """애널리스트 리포트에 특화된 문서 압축기"""
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[str, Document], str]] = None,
    ) -> "ReportLLMChainExtractor":
        """LLM으로부터 초기화"""
        
        def _get_input(query: str, doc: Document) -> dict:
            """쿼리와 문서 내용을 LLMChain 입력 형식으로 변환"""
            return {
                "query": query,
                "text": doc.page_content
            }
            
        _prompt = PromptTemplate(
            template="""다음 리포트 내용에서 사용자의 질문과 관련된 핵심 정보만 추출하여 간단히 요약해주세요.
            리포트의 element_type에 따라 다음과 같이 처리해주세요:
            
            1. text인 경우: 
               - 실적, 투자의견, 목표주가 등 정량적 데이터 위주로 추출
               - 분석의 핵심 근거나 중요 전망 포함
            
            2. image인 경우:
               - 차트/그래프가 보여주는 추세나 패턴 설명
               - 주요 수치들의 변화 내용 명시
               
            중요 지침:
            1. 사용자 질문과 무관한 내용은 제외
            2. 시기별 데이터는 연도/분기 명시
            3. 관련성이 없는 경우 빈 문자열("") 반환
            4. 요약은 명확하고 구체적인 수치 중심으로
            
            사용자 질문: {query}
            
            리포트 내용:
            {text}
            
            관련 정보 추출:""",
            input_variables=["query", "text"]  # 변수명 변경
        )
        
        _get_input = _get_input if get_input is None else get_input
        
        if _prompt.output_parser is not None:
            parser = _prompt.output_parser
        else:
            parser = StrOutputParser()
            
        llm_chain = _prompt | llm | parser
        return cls(llm_chain=llm_chain, get_input=_get_input)

class SearchDebugHandler(BaseCallbackHandler):
    """검색 과정의 디버깅을 위한 콜백 핸들러"""
    
    def on_chain_end(self, outputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        if hasattr(outputs, "filter") or hasattr(outputs, "query"):
            print("\n" + "="*50)
            print("🔍 구조화된 쿼리 생성 결과")
            print("="*50)
            filter_condition = getattr(outputs, "filter", None)
            query = getattr(outputs, "query", None)
            print(f"필터 조건: {filter_condition}")
            print(f"검색 쿼리: {query}")
            print("="*50 + "\n")

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> Any:
        print("\n" + "="*50)
        print("🚀 문서 검색 시작")
        print("="*50)
        print(f"검색 쿼리: {query}")
        print("="*50 + "\n")

    def on_retriever_end(self, documents: Sequence[Document], **kwargs: Any) -> Any:
        print("\n" + "="*50)
        print(f"✅ 검색 완료 - {len(documents)}개 문서 발견")
        print("="*50)
        for i, doc in enumerate(documents, 1):
            print(f"\n📄 문서 {i}:")
            print(f"내용: {doc.page_content}")
            print(f"메타데이터: {doc.metadata}")
        print("="*50 + "\n")

    def on_retriever_error(self, error: BaseException, **kwargs: Any) -> Any:
        print("\n" + "="*50)
        print("❌ 검색 오류 발생")
        print("="*50)
        print(f"오류 내용: {str(error)}")
        print("="*50 + "\n")

class CompressionDebugHandler(BaseCallbackHandler):
    """압축 과정의 디버깅을 위한 콜백 핸들러"""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print("\n" + "="*50)
        print("🔍 문서 압축 시작")
        print("="*50)
        print(f"입력 프롬프트: {prompts[0][:200]}...")
        print("="*50 + "\n")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        print("\n" + "="*50)
        print("✅ 문서 압축 완료")
        print("="*50)
        if hasattr(response, 'generations'):
            text = response.generations[0][0].text if response.generations else "응답 없음"
            print(f"압축 결과: {text[:200]}...")
        else:
            print("응답 형식이 예상과 다릅니다.")
        print("="*50 + "\n")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        print("\n" + "="*50)
        print("❌ 문서 압축 중 오류 발생")
        print("="*50)
        print(f"오류 내용: {str(error)}")
        print("="*50 + "\n")

class RetrievalManager:
    """검색기 관리 클래스"""
    
    def __init__(self, vectorstore, llm: Optional[ChatOpenAI] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vectorstore = vectorstore
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.query_rewriter = ReportQueryRewriter(self.llm)

        # 분석용 프롬프트 템플릿 초기화
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""당신은 금융 분석 전문가입니다. 애널리스트 리포트를 분석하여 
            통찰력 있는 답변을 제공합니다. 다음 지침을 따라 분석해주세요:
            
            1. 시기별 데이터 변화를 명확하게 설명 (분기별/연도별)
            2. 핵심 수치와 지표를 구체적으로 언급
            3. 변화의 주요 원인과 영향 요소 분석
            4. 향후 전망이나 예측 정보 포함 (있는 경우)
            5. 리포트의 핵심 내용만 추출하여 간단명료하게 답변"""),
            HumanMessagePromptTemplate.from_template("""
            질문: {query}
            
            관련 리포트 내용:
            {content}
            
            위 내용을 바탕으로 질문에 대한 전문가적인 답변을 제공해주세요.""")
        ])
        
        # 분석 체인 생성
        self.analysis_chain = self.analysis_prompt | self.llm


    # def create_self_query_retriever(self, k: int = 4) -> SelfQueryRetriever:
    #     """Self Query Retriever 생성"""
    #     try:
    #         prompt = get_query_constructor_prompt(
    #             document_contents=REPORT_DOCUMENT_DESCRIPTION,
    #             attribute_info=METADATA_FIELD_INFO,
    #             examples=SEARCH_EXAMPLES,
    #             allowed_comparators=["eq", "ne", "gt", "gte", "lt", "lte"],
    #             allowed_operators=["and", "or"]
    #         )

    #         parser = StructuredQueryOutputParser.from_components(
    #             allowed_comparators=["eq", "ne", "gt", "gte", "lt", "lte"],
    #             allowed_operators=["and", "or"]
    #         )

    #         query_constructor = prompt | self.llm | parser

    #         retriever = SelfQueryRetriever(
    #             query_constructor=query_constructor,
    #             vectorstore=self.vectorstore,
    #             structured_query_translator=ChromaTranslator(),
    #             search_kwargs={"k": k}
    #         )
            
    #         return retriever

    #     except Exception as e:
    #         error_msg = f"Self-query Retriever 생성 오류: {str(e)}"
    #         self.logger.error(error_msg)
    #         raise Exception(error_msg)

    def get_retriever_results(self, query: str, filter: Optional[Dict[str, Any]] = None, k: int = 4) -> dict:
        try:
            print(f"\n=== 검색 및 압축 프로세스 시작 ===")
            print(f"입력 쿼리: {query}")
            print(f"필터 조건: {filter}")
            
            # 1. 쿼리 재작성
            enhanced_query = self.query_rewriter.rewrite_query(query)
            # 쿼리에서 양쪽 따옴표 제거
            enhanced_query = enhanced_query.strip('"')
            print(f"\n재작성된 쿼리: {enhanced_query}")
            
            # 2. 검색 및 압축 설정
            compressor = ReportLLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": k, "filter": filter}
                )
            )
            
            # 3. 검색 및 압축 수행
            compressed_docs = compression_retriever.get_relevant_documents(
                enhanced_query,
                callbacks=[SearchDebugHandler(), CompressionDebugHandler()]
            )
            
            if not compressed_docs:
                print("\n❌ 검색 결과 없음")
                return {}
            
            # 시간순 정렬
            sorted_docs = sorted(
                compressed_docs,
                key=lambda x: (x.metadata.get("year", 0), x.metadata.get("month", 0)),
                reverse=True
            )
            
            # 4. 문서 내용 결합
            combined_content = "\n\n".join(
                f"[{doc.metadata.get('year', '알 수 없음')}년 {doc.metadata.get('month', '알 수 없음')}월]\n{doc.page_content.strip()}"
                for doc in sorted_docs 
                if doc.page_content.strip()
            )
            
            # 5. LLM을 통한 분석 및 답변 생성
            print("\n=== LLM 분석 시작 ===")
            analysis_response = self.analysis_chain.invoke({
                "query": query,
                "content": combined_content
            })
            print("\n✅ LLM 분석 완료")
            
            return {
                "analysis": analysis_response.content,
                "raw_content": combined_content
            }
            
        except Exception as e:
            error_msg = f"검색 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)