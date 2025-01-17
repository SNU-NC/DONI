from typing import Optional, List, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from ..utils.ticker_finder import TickerFinder
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate

class PeerInput(BaseModel):
    company: str = Field(description="회사명 (예: 삼성전자, Apple)")

class PeerBaseTool(BaseTool):
    name: str
    description: str
    args_schema: type[PeerInput] = PeerInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 기본 LLM, 프롬프트, 파서 설정
        self._llm = ChatOpenAI(temperature=0)
        
        response_schemas = [
            ResponseSchema(name="answer", description="사용자의 질문에 대한 답변, 파이썬 리스트 형식이어야 함."),
        ]
        self._output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self._output_parser.get_format_instructions()
        
        self._prompt_template = PromptTemplate(
            template="answer the users question as best as possible.\n{format_instructions}\n{question}",
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )

    def get_ticker_data(self, company: str) -> tuple:
        """회사의 티커 정보와 yfinance Ticker 객체를 반환합니다."""
        ticker_symbol = TickerFinder.get_ticker(company)
        if ticker_symbol is None:
            return None, None
        return ticker_symbol, yf.Ticker(ticker_symbol)

    def find_peer(self, company: str) -> List[str]:
        """경쟁사 목록을 찾습니다."""
        chain = self._prompt_template | self._llm | self._output_parser
        return chain.invoke({
            "question": f"{company}와 사업구조가 비슷하고, 같은 산업 혹은 섹터에 속한 경쟁사는?"
            "(코스피, 뉴욕거래소 등 상장된 회사만 찾으세요. 반드시 회사명만 출력해주세요.)"
        })

class PeerPERTool(PeerBaseTool):
    name: str = "peer_per_analysis_tool"
    description: str = "기업과 동종 업계의 Peer Group PER을 분석합니다"
    args_schema: type[PeerInput] = PeerInput
    
    def _run(self, company: str) -> Optional[dict]:
        peer_list = self.find_peer(company)['answer']
        peer_pers = {}
        
        for peer in peer_list:
            ticker_symbol, ticker = self.get_ticker_data(peer)
            if ticker is None:
                continue
                
            earning_ttm = sum(ticker.quarterly_income_stmt.loc['Net Income Common Stockholders'][:4])
            per = ticker.info["marketCap"]/earning_ttm
            
            if per < 0:
                continue
                
            if ".KS" not in ticker_symbol:
                per *= 0.7
                
            peer_pers[peer] = per
            
        if not peer_pers:
            return None
            
        return {
            "Peer PERs": peer_pers,
            "Peer list": peer_list,
            "Average Peer PER": sum(peer_pers.values()) / len(peer_pers)
        }
    
    def _arun(self, company: str):
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")

class PeerPBRTool(PeerBaseTool):
    name: str = "peer_pbr_analysis_tool"
    description: str = "기업과 동종 업계의 Peer Group PBR을 분석합니다"
    args_schema: type[PeerInput] = PeerInput
    
    def _run(self, company: str) -> Optional[dict]:
        peer_list = self.find_peer(company)['answer']
        peer_pbrs = {}
        
        for peer in peer_list:
            ticker_symbol, ticker = self.get_ticker_data(peer)
            if ticker is None:
                continue
                
            equity = ticker.quarterly_balance_sheet.loc['Stockholders Equity'][0]
            pbr = ticker.info["marketCap"]/equity
            
            if pbr < 0:
                continue
                
            if ".KS" not in ticker_symbol:
                pbr *= 0.7
                
            peer_pbrs[peer] = pbr
            
        if not peer_pbrs:
            return None
            
        return {
            "Peer PBRs": peer_pbrs,
            "Peer list": peer_list,
            "Average Peer PBR": sum(peer_pbrs.values()) / len(peer_pbrs)
        }
    
    def _arun(self, company: str):
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")