import logging
from typing import Sequence,  Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import (
    chain as as_runnable,
)
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, FunctionMessage, HumanMessage, BaseMessage , AIMessage
from plan.output_parser import LLMCompilerPlanParser
from plan.scheduler import schedule_tasks
from tools.analyze.report_agent.report_agent_Tool import ReportAgentTool
from tools.financialTerm.fin_knowledge_tools import get_fin_tool
from tools.extractor.query_processor_tool import get_query_processor_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import RunnableBranch
from datetime import datetime
from tools.is_target_tool import is_valid_query
from tools.planKB.plan_store import PlanStore
from langchain_openai import OpenAIEmbeddings

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')

class Planner:
    def __init__(self, llm: BaseChatModel, llm_mini: BaseChatModel, llm_clova: BaseChatModel, llm_for_candidates: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate):
        self.llm = llm
        self.tools = tools
        self.base_prompt = base_prompt
        self.planner = self._create_planner()
        self.llm_for_candidates = llm_for_candidates
        self.llm_mini = llm_mini
        self.llm_clova = llm_clova
        self.fin_tool = get_fin_tool(self.llm_mini)
        self.query_processor_tool = get_query_processor_tool(self.llm, self.llm_clova)
        self.report_agent_tool = ReportAgentTool()
        self.plan_store = PlanStore(OpenAIEmbeddings(
                model="text-embedding-3-small",  # 또는 "text-embedding-3-large"
                dimensions=1536  # 차원 수 지정 가능
                ))
        #self.plan_and_scheduler = self._create_plan_and_schedule()

    def _create_planner(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
            
        tool_descriptions = "\n".join(
            f"{i+1}. {tool.description}\n"
            for i, tool in enumerate(self.tools)
        )
        replanner_prompt = self.base_prompt.partial(
            
            replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you SHOULD NEVER repeat the actions that are already executed in the Previous Plan.\n"
            ' - You must continue the task index from the end of the previous one. Do not repeat task indices.\n',
            priority_of_tool_usage="""
            사용자 질문에 연도가 존재하지 않다면, 검색 연도를 2023년으로 도구들에게 질의하세요.
            사용자 질문이 사업 보고서에 바로 존재할만한 기업의 특수한 내용이라면 최대한 원본 쿼리를 combined_financial_report_search에 질의해보세요.
            만약 금융 함수 계산이 목적이거나 대소 비교와 같은 내용은 해당 문제를 풀기 위한 작은 문제로 분할하여 도구들에게 작업을 할당하세요.
            주식 가치평가, 동종업계 비교 질의, 주가 변동원인 분석 질문이라면 위 과정을 무시하고 아래 도구 사용법을 참고하세요
            - 주식 가치평가는 CombinedAnalysisTool를 사용하세요
            - 동종업계에 대한 정보는 SameSectorAnalyzer을 사용하세요
            - 주가 변동원인 분석은 StockAnalyzerTool을 사용하세요

            - 각 도구의 특성에 맞게 도구를 선택해야 합니다.
            - 이전 실행 결과를 참고하여 우선순위대로 계획을 짜세요 
            """,
            planning_candidates="",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
            time =current_time
        )

        def wrap_messages(state: list):
            """메시지를 딕셔너리로 감싸기"""
            return {"messages": state}

        def get_planning_candidates(state: Dict[str, Any]) -> Dict[str, Any]:
            """Planning candidates 생성"""
            try:
                # 1. messages 형식 검증 및 변환
                if isinstance(state["messages"], str):
                    messages = [HumanMessage(content=state["messages"])]
                elif isinstance(state["messages"], list):
                    if not state["messages"]:
                        raise ValueError("Empty messages list")
                    if isinstance(state["messages"][-1], BaseMessage):
                        messages = state["messages"]
                    else:
                        messages = [HumanMessage(content=msg) if isinstance(msg, str) else msg 
                                  for msg in state["messages"]]
                else:
                    raise ValueError(f"Unexpected messages type: {type(state['messages'])}")

                # 2. 쿼리 추출
                query = messages[-1].content
                
                # 3. plan_store 사용
                similar_examples = self.plan_store.get_similar_examples(query)
                print("logging for similar_examples")
                print(similar_examples)

                # 4. extra_info가 있다면 원본 메시지에 추가
                for example in similar_examples:
                    if example.extra_info:
                        # 기존 메시지 내용에 extra_info 추가
                        original_content = messages[-1].content
                        agmentedContent=f"{original_content}\n\n[참고사항] {example.extra_info}"

                        messages[-1].content = agmentedContent
                
                # 5. 결과 포맷팅
                planning_candidates = "\n\n".join([
                    "사용자의 질문과 유사하다면 다음 계획 예시를 참고하세요:\n" +
                    "<EXAMPLE>\n" +
                    f"Query: {example.query}\n" + 
                    "\n".join([f"- {step}" for step in example.plan.steps]) +
                    "\n</EXAMPLE>"
                    for i, example in enumerate(similar_examples)
                ])
                
                print("logging for planning_candidates Results")
                print(planning_candidates)
                return {
                    "messages": messages,  # 변환된 messages 반환
                    "planning_candidates": planning_candidates
                }
                
            except Exception as e:
                logging.error(f"Error generating planning candidates: {e}")
                return {
                    "messages": state.get("messages", []),
                    "planning_candidates": ""
                }

        # 기본 계획 생성을 위한 체인
        planning_chain = (
            RunnablePassthrough.assign(
                num_tools=lambda _: len(self.tools) + 1,
                tool_descriptions=lambda _: tool_descriptions,
                time=lambda _: current_time,
                priority_of_tool_usage=lambda _: """
            사용자 질문에 연도가 존재하지 않다면, 검색 연도를 2023년으로 도구들에게 질의하세요.
            
            ### combined_financial_report_search 도구 사용 시 주의사항
            - 기업당 1회만 호출이 가능하므로 연도가 다양해도 최대한 자세하게 필요한 정보를 한번에 요청하세요
            - 예시) "2021,2022,2023년 연구개발비 추이를 알려주세요" 
            - 여러 기업을 비교할 때도 기업별로 1회씩만 호출하여 여러 연도의 데이터를 한번에 요청하세요
            - 예시) 기업A: "2021,2022,2023년 연구개발비", 기업B: "2021,2022,2023년 연구개발비"
            - 연도와 필요한 계정명을 사용자 쿼리에서 파악하여 이를 포함하세요
            
            ### 다른 도구 사용 시 주의사항
            - 금융 함수 계산이나 대소 비교가 필요한 경우 작은 문제로 분할하여 처리하세요
            - 주가/주식/환율 데이터 검색은 market_data_tool을 사용하세요
            - 주가 가치평가는 CombinedAnalysisTool를 사용하세요
            - 동종업계에 대한 정보는 SameSectorAnalyzer을 사용하세요
            - 주가 변동원인 분석은 StockAnalyzerTool을 사용하세요
            
            - 각 도구의 특성에 맞게 도구를 선택하고 이전 실행 결과를 참고하여 우선순위대로 계획을 짜세요
                """
            )
            | self.base_prompt
        )

        def should_replan(state: list):
            print("$$$$$$$$$$$$$ State 를 확인하겠습니다 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(state)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            return isinstance(state[-1], SystemMessage)

        def wrap_and_get_last_index(state: list):
            next_task = 0
            for message in state[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = message.additional_kwargs["idx"] + 1
                    break
            state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
            return {"messages": state}
        return (
            RunnableBranch(
                (should_replan, wrap_and_get_last_index | replanner_prompt),
                wrap_messages | RunnableLambda(get_planning_candidates) | planning_chain,
            )
            | self.llm
            | LLMCompilerPlanParser(tools=self.tools)
        )

    def create_plan_and_schedule(self, state: Dict[str, Any]):
        """plan_and_schedule 함수를 생성하여 반환"""
        planner = self.planner
        llm = self.llm
        llm_mini = self.llm_mini
        def is_in_report_agent(company_name: str) -> bool:
            # 리포트 에이전트에서 처리할 수 없는 기업인지 확인 
                # 리포트 에이전트가 처리할 수 있는 기업 목록
            report_companies = {
                "SK하이닉스": "000660",
                "대한항공": "003490",
                "대한항공우": "003495",
                "포스코퓨처엠": "003670",
                #"현대차": "005380",
                "SK텔레콤": "017670",
                "삼성전자": "005930",
                "두산에너빌리티": "034020",
                "NAVER": "035420",
                "카카오": "035720",
                "넷마블": "251270",
                "LG에너지솔루션": "373220"
            }

            return company_name in report_companies

        @as_runnable
        def plan_and_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            try:
                # 1. 원본 쿼리 ( = user query )
                original_query = messages[0].content

                # replan_count가 없을 때만 fin_tool 실행
                input_query = original_query
                if "replan_count" not in state:
                    input_query = self.fin_tool.invoke({"query": original_query})
                    process_result = self.query_processor_tool.invoke({"query": input_query})
                    print("process_result:", process_result)
                    input_query = process_result["input_query"]
                    metadata = process_result["metadata"]
                messages[0].content = input_query
                logging.info(f"원본 쿼리: {original_query}")
                logging.info(f"FinTool 사용 후의 쿼리: {input_query}")
                
                # 리포트 에이전트에서 처리할 수 없는 기업인지 확인 
                is_report_can = is_in_report_agent(metadata["companyName"])
                print("is_report_can:", is_report_can)

                # report_agent_use 판단
                if is_valid_query(original_query) and is_report_can:
                    logging.info("report_agent_use 판단 시작")
                    result_message = self.report_agent_tool.invoke({"query": original_query, "metadata": metadata})
                    logging.info("report_agent_tool 사용 후 바로 종료합니다.")
                    # 상위 레벨(should_continue)에서 이 값을 확인해 종료 처리
                    return {
                        "messages": [result_message],
                        "replan_count": state.get("replan_count", 0),
                        "report_agent_use": True
                    }

                try:

                    initial_tasks = list(planner.stream(messages))
                    if not initial_tasks:
                        logging.warning("초기 태스크 생성 실패: 빈 태스크 리스트")
                        return {"messages": [], "replan_count": state.get("replan_count", 0)}
                except Exception as e:
                    logging.error(f"초기 태스크 생성 중 오류 발생: {e}")
                    return {"messages": [], "replan_count": state.get("replan_count", 0)}

                # 2. task 스케줄링
                try:
                    if initial_tasks:
                        logging.info(f"스케줄링할 태스크 수: {len(initial_tasks)}")
                        scheduled_tasks, new_task_results = schedule_tasks.invoke({
                            "messages": messages,
                            "tasks": initial_tasks
                        })
                        # state의 task_results에 새로운 결과 추가
                        current_task_results = state.get("task_results", [])
                        current_task_results.extend(new_task_results)
                    else:
                        logging.warning("스케줄링할 태스크가 없음")
                        scheduled_tasks = []
                        current_task_results = state.get("task_results", [])
                except Exception as e:
                    logging.error(f"태스크 스케줄링 중 오류: {e}")
                    scheduled_tasks = []
                    current_task_results = state.get("task_results", [])

                # 6. replan count 관리
                if "replan_count" not in state:
                    state["replan_count"] = 0
                    logging.info("replan_count 초기화")
                state["replan_count"] = state["replan_count"] + 1
                logging.info(f"replan_count 증가: {state['replan_count']}")

                return {
                    "messages": scheduled_tasks, 
                    "replan_count": state["replan_count"],
                    "task_results": current_task_results  # 누적된 task_results 반환
                }

            except Exception as e:
                logging.error(f"plan_and_schedule 전체 실행 중 예기치 않은 오류: {e}")
                return {
                    "messages": [], 
                    "replan_count": state.get("replan_count", 0),
                    "task_results": state.get("task_results", [])  # 기존 task_results 유지
                }

        return plan_and_schedule
