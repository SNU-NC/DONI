import yfinance as yf
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from tools.analyze.report_agent.tools.report_agent_utils import get_ticker

response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변, 파이썬 리스트 형식이어야 함."),
    ]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 출력 형식 지시사항을 파싱합니다.
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    # 사용자의 질문에 최대한 답변하도록 템플릿을 설정합니다.
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    # 입력 변수로 'question'을 사용합니다.
    input_variables=["question"],
    # 부분 변수로 'format_instructions'을 사용합니다.
    partial_variables={"format_instructions": format_instructions},
)

def find_peer(company: str, llm) -> list[str]:
    prompt = PromptTemplate(
    # 사용자의 질문에 최대한 답변하도록 템플릿을 설정합니다.
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    # 입력 변수로 'question'을 사용합니다.
    input_variables=["question"],
    # 부분 변수로 'format_instructions'을 사용합니다.
    partial_variables={"format_instructions": format_instructions},
    )
    chain = prompt | llm | output_parser  # 프롬프트, 모델, 출력 파서를 연결
    peer_list = chain.invoke({"question": f"{company}와 사업구조가 비슷하고, 같은 산업 혹은 섹터에 속한 경쟁사는?"
                              "(코스피, 뉴욕거래소 등 상장된 회사만 찾으세요. 반드시 회사명만 출력해주세요.)"})
    return peer_list

def find_peer_PERs_tool(company: str, state, llm) -> None:
    """기업과 동종 업계의 Peer Group PER 평균을 찾습니다."""
    peer_list = find_peer(company, llm)['answer']

    # print(f"필터링 전 경쟁사 리스트: {peer_list}")
    
    peer_pers = {}
    for peer in peer_list:
        # print(f"경쟁사: {peer}")
        ticker = None
        ticker = get_ticker(peer)
        # print(f"경쟁사 티커: {ticker}")
        if ticker is None:
            continue
        elif ".KS" in ticker:
            ticker = yf.Ticker(ticker)
            earning_ttm = 0
            try:
                for i in range(4):
                    earning_ttm += ticker.quarterly_income_stmt.loc['Net Income Common Stockholders'][i]
            except:
                print(f"{peer}의 Net Income Common Stockholders 문제 발생")
                continue
            trailingPERttm = ticker.info.get("marketCap")/earning_ttm
            # print(f"경쟁사 PER: {trailingPERttm}")
            if trailingPERttm <0 :
                continue
            peer_pers[peer] = trailingPERttm
        else:
            ticker = yf.Ticker(ticker)
            earning_ttm = 0
            try:
                for i in range(4):
                    earning_ttm += ticker.quarterly_income_stmt.loc['Net Income Common Stockholders'][i]
            except:
                print(f"{peer}의 Net Income Common Stockholders 문제 발생")
                continue
            trailingPERttm = ticker.info.get("marketCap")/earning_ttm*0.7 # 외국 주식의 경우 PER을 30% 할인
            # print(f"경쟁사 PER: {trailingPERttm}")
            if trailingPERttm <0 :
                continue
            if np.isnan(trailingPERttm):
                continue
            peer_pers[peer] = trailingPERttm

    valid_peer_pers = {}
    # print(f"현재 PER: {state.PER}")

    if state.PER < 0:
        valid_peer_pers = peer_pers
    else:
        for key, value in peer_pers.items():
            if value < 0.1 * state.PER or value > 10 * state.PER:
                continue
            else:
                valid_peer_pers[key] = value
    print(valid_peer_pers)
    # print(f"필터링 후 경쟁사 수: {len(valid_peer_pers)}")                    
    average_peer_per = sum(valid_peer_pers.values()) / len(valid_peer_pers)

    valid_peer_list = []
    for key, value in valid_peer_pers.items():
        valid_peer_list.append(key)
    # print(f"필터링 후 경쟁사 리스트: {valid_peer_list}")

    return valid_peer_list, average_peer_per

def find_PER_tool(ticker: str) -> float:
    """기업의 현재 PER(TTM)를 찾습니다."""

    ticker = yf.Ticker(ticker + ".KS")
    earning_ttm = 0
    for i in range(4):
        earning_ttm += ticker.quarterly_income_stmt.loc['Net Income Common Stockholders'][i]
    trailingPERttm = ticker.info["marketCap"]/earning_ttm
    return trailingPERttm
