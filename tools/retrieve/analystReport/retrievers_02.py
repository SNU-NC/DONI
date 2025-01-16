import json
import asyncio
from langchain_community.document_loaders import AsyncChromiumLoader
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

class WebScrapeRetriever:
    def __init__(self):
        self.kospi_mapping = self._load_mapping()
        self.gpt_4o = ChatOpenAI(
            model="chatgpt-4o-latest",
            temperature=0,
        )
        self.gpt_4o_mini = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
        )
    
    def _load_mapping(self) -> dict:
        """Load KOSPI mapping data"""
        with open('tools/retrieve/data/kospi_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    async def _scrape_report_detail(self, detail_url: str, query: str, basic_metadata: dict) -> Dict:
        """Scrape detailed report information"""
        try:
            loader = AsyncChromiumLoader([detail_url])
            html = await loader.aload()
            
            if not html:
                return {}
            
            soup = BeautifulSoup(html[0].page_content, 'html.parser')
            
            # 투자의견과 목표가 추출
            view_info = soup.find('div', class_='view_info')
            target_price = ''
            opinion = ''
            
            if view_info:
                price_elem = view_info.find('em', class_='money')
                if price_elem:
                    target_price = price_elem.text.strip()
                
                opinion_elem = view_info.find('em', class_='coment')
                if opinion_elem:
                    opinion = opinion_elem.text.strip()
            
            # 본문 미리보기 추출
            content = ""
            view_cnt = soup.find('td', class_='view_cnt')
            if view_cnt:
                paragraphs = view_cnt.find_all(['p'])
                content = '\n'.join([p.text.strip() for p in paragraphs if p.text.strip()])
            
            # 메타데이터 구성
            metadata = {
                **basic_metadata,  # 기본 메타데이터 (종목명, 제목, 증권사 등)
                'url': detail_url,
                'target_price': target_price,
                'investment_opinion': opinion
            }
            
            EXTRACT_PROMPT = PromptTemplate(
                template="""다음 문서와 테이블 정보에서 사용자의 질문과 관련된 정보를 찾아 형식에 맞게 추출하세요.

                핵심 규칙:
                1. 문서의 내용이 질문과 관련이 있는지 판단하세요
                2. 관련이 있다면 다음 두 가지를 명확히 작성하세요:
                   - 정보가 필요했던 부분: 사용자 질문의 어떤 부분에 대한 정보인지
                   - 참고한 내용: 문서에서 찾은 실제 정보 (연도/기수 정보 포함)
                3. 관련이 없다면 빈 객체를 반환하세요
                4. 연도나 기수 정보가 있는 경우:
                   - 해당 연도/기수의 정보를 우선적으로 추출
                   - 이전 연도와의 비교 정보가 있다면 함께 포함
                   - 시점이 다른 정보는 명확히 구분하여 표시
                5. 연도나 기수 정보가 없는 경우:
                   - 사용자 쿼리에 없다면 최근 연도 기준으로 추출하세요
                   - 사용자 쿼리에 연도가 있지만 찾은 데이터에 없다면 연도 정보가 불확실함을 명확히 표시하세요

                예시 응답:
                {{"needed_information": "주가전망",
                  "referenced_content": "젊은(=최신 트랜드) UGC의 증가는 검색서비스 품질 향상으로 이어져 당사의 주요 매출원인 검색매출 증가에도 기여할 수 있을 것으로 기대됨, 출처: [유안타증권]24.12.30",
                }}
                또는
                {{"needed_information": "",
                  "referenced_content": "",
                }}

                ### 검색 결과
                사용자 질문: {query}

                문서 내용:
                {content}

                참고할 문서 메타데이터:
                {metadata}

                검색 결과:""",
                input_variables=["query", "content", "metadata"]
            )

            # GPT-4를 사용하여 needed_information 생성
            if content:
                _prompt = EXTRACT_PROMPT
                extraction_chain = _prompt | self.gpt_4o_mini | StrOutputParser()
                needed_information = extraction_chain.invoke({
                    "query": query,
                    "content": content,
                    "metadata": metadata
                })
            else:
                needed_information = "문서 정보"
            
            return {
                "target_price": target_price,
                "investment_opinion": opinion,
                "content": content,
                "needed_information": needed_information
            }
        except Exception as e:
            print(f"상세 정보 스크래핑 중 오류 발생: {str(e)}")
            return {}

    async def _scrape_data(self, search_keyword: str, query: str) -> List[Dict]:
        """Scrape data from Naver finance"""
        ticker = None
        for company, code in self.kospi_mapping.items():
            if search_keyword == company:
                ticker = code.split('.')[0]  # .KS 제거
                break
        
        if not ticker:
            return []

        url = f"https://finance.naver.com/research/company_list.naver?keyword=&brokerCode=&writeFromDate=&writeToDate=&searchType=itemCode&itemCode={ticker}&x=17&y=19"
        
        try:
            loader = AsyncChromiumLoader([url])
            html = await loader.aload()
            
            if not html:
                return []
                
            soup = BeautifulSoup(html[0].page_content, 'html.parser')
            research_data = []
            
            table = soup.find('table', {'class': 'type_1'})
            if table:
                rows = table.find_all('tr')
                count = 0
                for row in rows:
                    if count >= 5:
                        break
                        
                    cols = row.find_all('td')
                    if len(cols) >= 6 and not row.has_attr('class'):
                        report_detail = cols[1].find('a')
                        download_url = cols[3].find('a').get('href')
                        # print(f"download_url: {download_url}")
                        if report_detail and report_detail.get('href'):
                            detail_url = "https://finance.naver.com/research/" + report_detail.get('href')
                            
                            # 기본 메타데이터 구성
                            basic_metadata = {
                                'company': cols[0].text.strip(),
                                'title': cols[1].text.strip(),
                                'broker': cols[2].text.strip(),
                                'date': cols[4].text.strip(),
                                'views': cols[5].text.strip(),
                            }
                            
                            # 상세 페이지 스크래핑 - query와 기본 메타데이터 전달
                            detail_info = await self._scrape_report_detail(detail_url, query, basic_metadata)
                            
                            research_item = {
                                '종목명': basic_metadata['company'],
                                '제목': basic_metadata['title'],
                                '증권사': basic_metadata['broker'],
                                '작성일': basic_metadata['date'],
                                '조회수': basic_metadata['views'],
                                '링크': download_url,
                                '목표가': detail_info.get('target_price', ''),
                                '투자의견': detail_info.get('investment_opinion', ''),
                                '본문': detail_info.get('content', ''),
                                'needed_information': detail_info.get('needed_information', '문서 정보')
                            }
                            # print(f"research_item: {research_item}")
                            
                            if research_item['종목명'] and not research_item['종목명'].isspace():
                                research_data.append(research_item)
                                count += 1
            
            return research_data
                
        except Exception as e:
            print(f"스크래핑 중 에러 발생: {str(e)}")
            return []

    def _create_documents(self, results: List[Dict]) -> List[Document]:
        """Create Document objects from scraped results"""
        documents = []
        for item in results:
            # 메타데이터와 컨텐츠 분리
            metadata = {
                'company': item['종목명'],
                'title': item['제목'],
                'broker': item['증권사'],
                'date': item['작성일'],
                'views': item['조회수'],
                'url': item['링크'],
                'target_price': item['목표가'],
                'investment_opinion': item['투자의견'],
                'needed_information': item['needed_information']  # 추가된 부분
            }
            
            # 본문을 content로 사용
            content = item['본문']
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def _format_documents(self, query: str, documents: List[Document]) -> Dict:
        """Format documents into a structured response"""
        from tools.retrieve.financialReport.prompts import output_parser

        if not documents:
            return {
                "output": "검색 결과를 찾을 수 없습니다.",
                "key_information": []
            }
        
        documents = documents[:5]
        
        # 도큐먼트 포맷팅 (combined_content용)
        formatted_content = []
        for idx, doc in enumerate(documents, 1):
            formatted_content.append(
                f"[{idx}번째 리포트]\n"
                f"종목명: {doc.metadata['company']}\n"
                f"제목: {doc.metadata['title']}\n"
                f"증권사: {doc.metadata['broker']}\n"
                f"작성일: {doc.metadata['date']}\n"
                f"조회수: {doc.metadata['views']}\n"
                f"목표가: {doc.metadata['target_price']}\n"
                f"투자의견: {doc.metadata['investment_opinion']}\n"
                f"링크: {doc.metadata['url']}\n"
                f"관련성: {doc.metadata['needed_information']}\n"
                f"본문 미리보기:\n{doc.page_content}\n"
                f"{'-' * 50}"
            )
        
        combined_content = "\n".join(formatted_content)

        # GPT 요약을 위한 문서 포맷팅
        summary_docs = []
        for doc in documents:
            summary_docs.append(
                f"[리포트 정보]\n"
                f"제목: {doc.metadata['title']}\n"
                f"증권사: {doc.metadata['broker']}\n"
                f"작성일: {doc.metadata['date']}\n"
                f"목표가: {doc.metadata['target_price']}\n"
                f"투자의견: {doc.metadata['investment_opinion']}\n"
                f"링크: {doc.metadata['url']}\n"
                f"관련성: {doc.metadata['needed_information']}\n"
                f"내용:\n{doc.page_content}\n"
                f"{'-' * 50}"
            )
        
        docs_text = "\n".join(summary_docs)

        # 최종 요약 프롬프트
        FINAL_SUMMARY_PROMPT = PromptTemplate(
    template="""주어진 검색 결과들을 바탕으로 사용자의 질문에 대한 명확한 답변을 작성해주세요.

            사용자 질문: {query}

            검색 결과:
            {search_results}

            답변은 다음 형식으로 작성해주세요:
            {{"output": "사용자 질문에 대한 상세한 답변 내용",
             "key_information": [
                {{"tool": "애널리스트 보고서",
                  "referenced_content": "답변 작성에 실제로 사용된 검색 결과 내용만 포함",
                  "title": "리포트 제목",
                  "broker": "증권사",
                  "target_price": "목표가",
                  "investment_opinion": "투자의견",
                  "link": "첨부된 링크"
                }},
                ...
             ]
            }}

            주의사항:
            - 각 referenced_content 항목은 output에서 언급된 내용과 직접적으로 연관되어야 합니다
            - key_information에는 답변 작성에 실제로 사용된 정보만 포함해야 합니다
            - key_information에 중복된 내용이 있으면 제외하세요.
            - 검색 결과에서 답변에 반영되지 않은 내용은 key_information에서 제외하세요
           """,
    input_variables=["query", "search_results"]
)

        # 최종 요약 체인 생성
        final_summary_chain = (
            FINAL_SUMMARY_PROMPT 
            | self.gpt_4o
            | output_parser
        )

        try:
            # 도큐먼트 메타데이터 기반으로 key_information 구성
            report_results = {
                "output": "",
                "key_information": []
            }

            for doc in documents:
                referenced_content = doc.page_content
                if referenced_content and referenced_content.strip():
                    metadata = doc.metadata
                    report_results["key_information"].append({
                        "tool": "애널리스트 보고서",
                        "needed_information": metadata.get("needed_information", "문서 정보"),
                        "referenced_content": referenced_content,
                        "title": metadata.get("title", "N/A"),
                        "broker": metadata.get("broker", "N/A"),
                        "target_price": metadata.get("target_price", "N/A"),
                        "investment_opinion": metadata.get("investment_opinion", "N/A"),
                        "link": metadata.get("url", "N/A")
                    })

            # 최종 요약 생성 및 결과 반환
            final_input = {
                "query": query,
                "search_results": docs_text  # summary_docs에서 생성한 텍스트 사용
            }
            
            results = final_summary_chain.invoke(final_input)
            
            return {
                "output": results['output'],
                "key_information": results.get('key_information', [])
            }

        except Exception as e:
            print(f"요약 생성 중 오류 발생: {str(e)}")
            return {
                "output": "요약 생성 중 오류가 발생했습니다.",
                "key_information": []
            }

    async def run(self, query: str, company: str) -> Dict:
        """Run the web scraping process"""
        try:
            if not company:
                return {
                    "output": "회사명을 입력해주세요.",
                    "key_information": []
                }
            
            results = await self._scrape_data(company, query)
            documents = self._create_documents(results)
            return self._format_documents(query, documents)
            
        except Exception as e:
            error_message = f"검색 중 오류 발생: {str(e)}"
            return {
                "output": error_message,
                "key_information": []
            }