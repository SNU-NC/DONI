import json
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
from config.prompts import _PLANNING_CANDIDATES_PROMPT
from plan.output_parser import LLMCompilerPlanParser
from plan.scheduler import schedule_tasks
from tools.analyze.report_agent.report_agent_Tool import ReportAgentTool
from tools.financialTerm.fin_knowledge_tools import get_fin_tool
from tools.extractor.query_processor_tool import get_query_processor_tool
from tools.extractor.quick_retriever_tool import QuickRetrieverTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import RunnableBranch
from datetime import datetime
from tools.is_target_tool import is_valid_query # report_agent 사용여부 판단
from tools.query_analyzer_tool import query_analyzer # quick_retriever_tool 사용여부, plan_and_schedule 사용여부 판단
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
        self.quick_retriever_tool = QuickRetrieverTool(self.llm)
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
            
            웹검색은 만능이지만 수치 데이터를 가져오는 목적으론 사용하지 마세요
            웹검색 도구는 통합 검색기를 부른다면 선택적으로 사용하세요
            계획 참고:
            1. 트랜드나 사업보고서에서 찾기 힘든 정보를 찾는데 사용하세요
            2. 부가적인 설명을 추가하고 싶을 때 사용하세요

            ### combined_financial_report_search 도구 사용 시 주의사항
            - 기업당 1회만 호출이 가능하므로 연도가 다양해도 최대한 자세하게 필요한 정보를 한번에 요청하세요
            - 예시) "2021,2022,2023년 연구개발비 추이를 알려주세요" 
            - 여러 기업을 비교할 때도 기업별로 1회씩만 호출하여 여러 연도의 데이터를 한번에 요청하세요
            - 예시) 기업A: "2021,2022,2023년 연구개발비", 기업B: "2021,2022,2023년 연구개발비"
            - 사용자의 의도를 최대한 담아주세요
            - 예시) 메리츠금융지주의 2023년 영업이익 중 손해보험업 부문이 차지하는 비율
            - 연도와 필요한 계정명을 사용자 쿼리에서 파악하여 이를 포함하세요
            - 사업보고서에는 ROE, ROA, 주당순이익(EPS) 등 주요 지표가 있을 수 있으니 해당 정보도 같이 검색해야합니다
            
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
        
        # @as_runnable
        # def plan_and_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
        #     messages = state["messages"]
        #     print("Plan*Scheduler 시작할 때의 replan Count를 체크하겠습니다. ", state.get("replan_count", 404))
        #     try:
        #         # 1. 원본 쿼리 ( = user query )
        #         original_query = messages[0].content

        #         # 2. replan_count가 없을 때만 fin_tool, query_processor_tool 실행
        #         input_query = original_query
        #         if "replan_count" not in state:
        #             input_query = self.fin_tool.invoke({"query": original_query})
        #             process_result = self.query_processor_tool.invoke({"query": input_query})
        #             print("process_result:", process_result)

        #             # query_analyzer_result로 판단
        #             query_analyzer_result = query_analyzer(process_result.get("input_query"))
        #             print("query_analyzer_result:", query_analyzer_result)

        #             # quick_retriever_tool 사용 필요하면
        #             if query_analyzer_result.get("quick_retriever_tool"):
        #                 quick_retriever_result = self.quick_retriever_tool.invoke(process_result)
        #                 print("quick_retriever_result:", quick_retriever_result)

        #                 # plan_and_schedule 사용 필요 없으면, join으로 메시지 보내기
        #                 if not query_analyzer_result.get("plan_and_schedule"):
        #                     # quick_retriever_result에서 유효한 정보를 얻었다면, 
        #                     # return 할때 quick_retriever_tool 을 message로 감싼 다음에 state에 message로 넘겨주기
        #                     print("quick_retriever_tool만 사용합니다.")
        #                     # 유효한 결과가 있는 경우
        #                     key_info = quick_retriever_result["key_information"][0]
        #                     # print("key_info:", key_info)
        #                     # FunctionMessage로 변환
        #                     retriever_message = FunctionMessage(
        #                         content= json.dumps(quick_retriever_result),
        #                         name="quick_retriever_tool",
        #                         additional_kwargs={
        #                             "tool": key_info["tool"],
        #                             "company": key_info["company"],
        #                             "referenced_content": key_info["referenced_content"],
        #                             "link": key_info["link"],
        #                             "idx": 1 # 첫번째 테스크로 처리
        #                         }
        #                     )
        #                     print("retriever_message:", retriever_message)
                            
        #                     # messages에 retriever_message를 추가해줘야 함
        #                     messages.append(retriever_message)  # 여기 추가

        #                     return {
        #                         "messages": messages,  
        #                         "replan_count": 1,
        #                         "key_information": [key_info],
        #                         "quick_retriever_message": retriever_message
        #                     }
                            
                        
        #                 # plan_and_schedule 사용 필요하면, quick_retriever_result 결과를 messages에 추가
        #                 else:
        #                     print("quick_retriever_tool과 plan_and_schedule 둘 다 사용합니다.")
        #                     key_info = quick_retriever_result["key_information"][0]
                            
        #                     # FunctionMessage로 변환
        #                     retriever_message = FunctionMessage(
        #                         content= json.dumps(quick_retriever_result),
        #                         name="quick_retriever_tool",
        #                         additional_kwargs={
        #                             "tool": key_info["tool"],
        #                             "company": key_info["company"],
        #                             "referenced_content": key_info["referenced_content"],
        #                             "link": key_info["link"],
        #                             "idx": 1
        #                         }
        #                     )
        #                     print("retriever_message:", retriever_message)

        #                     messages.append(retriever_message)  # quick_retriever 결과를 messages에 추가

        #                     # planner 실행
        #                     try:
        #                         initial_tasks = list(planner.stream(messages))
        #                         if initial_tasks:
        #                             logging.info(f"스케줄링할 태스크 수: {len(initial_tasks)}")
        #                             scheduled_tasks, new_task_results = schedule_tasks.invoke({
        #                                 "messages": messages,
        #                                 "tasks": initial_tasks
        #                             })
                                    
        #                             # current_task_results 초기화 및 누적
        #                             current_task_results = state.get("task_results", [])
        #                             current_task_results.extend(new_task_results)
                                    
        #                             # task_results에서 key_information 수집
        #                             all_key_information = [key_info]  # quick_retriever의 key_info 먼저 추가
        #                             for task_result in current_task_results:
        #                                 if isinstance(task_result, dict) and 'result' in task_result:
        #                                     result = task_result['result']
        #                                     if isinstance(result, dict) and 'key_information' in result:
        #                                         all_key_information.extend(result['key_information'])   

        #                             return {
        #                                 "messages": scheduled_tasks,
        #                                 "replan_count": 0,
        #                                 "key_information": all_key_information,
        #                                 "task_results": current_task_results,  # task_results 포함
        #                                 "quick_retriever_message": retriever_message
        #                             }
        #                         else:
        #                             return {
        #                                 "messages": messages,
        #                                 "replan_count": 0,
        #                                 "key_information": [key_info],
        #                                 "task_results": state.get("task_results", []),  # 기존 task_results 유지
        #                                 "quick_retriever_message": retriever_message
        #                             }
        #                     except Exception as e:
        #                         logging.error(f"Plan execution error: {e}")
        #                         return {
        #                             "messages": messages,
        #                             "replan_count": 0,
        #                             "key_information": [key_info],
        #                             "task_results": state.get("task_results", []),  # 기존 task_results 유지
        #                             "quick_retriever_message": retriever_message
        #                         }
                            

        #             # quick_retriever_tool은 필요하지 않고, plan_and_schedule만 필요하면 (O)
        #             else:
        #                 input_query = process_result.get("input_query")
        #                 print("input_query:", input_query)
        #                 metadata = process_result.get("metadata")
        #                 print("metadata:", metadata)

        #         input_query = process_result.get("input_query")
        #         print("input_query:", input_query)
        #         metadata = process_result.get("metadata")
        #         print("metadata:", metadata)

        #         messages[0].content = input_query
        #         logging.info(f"원본 쿼리: {original_query}")
        #         logging.info(f"FinTool 사용 후의 쿼리: {input_query}")
                
        #         # 리포트 에이전트에서 처리할 수 없는 기업인지 확인 
        #         is_report_can = is_in_report_agent(metadata["companyName"])
        #         print("is_report_can:", is_report_can)

        #         # report_agent_use 판단
        #         if is_valid_query(original_query) and is_report_can:
        #             logging.info("report_agent_use 판단 시작")
        #             result_message = self.report_agent_tool.invoke({"query": original_query, "metadata": metadata})
        #             logging.info("report_agent_tool 사용 후 바로 종료합니다.")
        #             # 상위 레벨(should_continue)에서 이 값을 확인해 종료 처리
        #             return {
        #                 "messages": [result_message],
        #                 "replan_count": state.get("replan_count", 1),
        #                 "report_agent_use": True
        #             }

        #         try:

        #             initial_tasks = list(planner.stream(messages))
        #             if not initial_tasks:
        #                 logging.warning("초기 태스크 생성 실패: 빈 태스크 리스트")
        #                 return {"messages": [], "replan_count": state.get("replan_count", 1)}
        #         except Exception as e:
        #             logging.error(f"초기 태스크 생성 중 오류 발생: {e}")
        #             return {"messages": [], "replan_count": state.get("replan_count", 1)}

        #         # 3. task 스케줄링
        #         try:
        #             if initial_tasks:
        #                 logging.info(f"스케줄링할 태스크 수: {len(initial_tasks)}")
        #                 scheduled_tasks, new_task_results = schedule_tasks.invoke({
        #                     "messages": messages,
        #                     "tasks": initial_tasks
        #                 })
        #                 # state의 task_results에 새로운 결과 추가
        #                 current_task_results = state.get("task_results", [])
        #                 current_task_results.extend(new_task_results)
        #             else:
        #                 logging.warning("스케줄링할 태스크가 없음")
        #                 scheduled_tasks = []
        #                 current_task_results = state.get("task_results", [])
        #         except Exception as e:
        #             logging.error(f"태스크 스케줄링 중 오류: {e}")
        #             scheduled_tasks = []
        #             current_task_results = state.get("task_results", [])

        #         # 4. replan count 관리
        #         if "replan_count" not in state:
        #             state["replan_count"] = 0
        #             print("replan_count 초기화", state["replan_count"])
        #         else : 
        #             state["replan_count"] = state["replan_count"] + 1
        #             print("replan_count 증가", state["replan_count"])
                

        #         return {
        #             "messages": scheduled_tasks, 
        #             "replan_count": state["replan_count"],
        #             "task_results": current_task_results  # 누적된 task_results 반환
        #         }

        #     except Exception as e:
        #         logging.error(f"plan_and_schedule 전체 실행 중 예기치 않은 오류: {e}")
        #         return {
        #             "messages": [], 
        #             "replan_count": state.get("replan_count", 1),
        #             "task_results": state.get("task_results", [])  # 기존 task_results 유지
        #         }
        @as_runnable
        def plan_and_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            print("Plan*Scheduler 시작할 때의 replan Count를 체크하겠습니다. ", state.get("replan_count", 404))
            
            try:
                # replan 시나리오 처리
                if "replan_count" in state:
                    state["replan_count"] += 1
                    
                    # 기존의 task_results와 key_information 유지
                    current_task_results = state.get("task_results", [])
                    current_key_information = state.get("key_information", [])
                    
                    # quick_retriever_message 처리
                    quick_retriever_message = state.get("quick_retriever_message")
                    if quick_retriever_message and quick_retriever_message not in messages:
                        messages.insert(1, quick_retriever_message)
                    
                    try:
                        initial_tasks = list(planner.stream(messages))
                        if initial_tasks:
                            scheduled_tasks, new_task_results = schedule_tasks.invoke({
                                "messages": messages,
                                "tasks": initial_tasks
                            })
                            
                            # 새로운 task_results를 기존 결과에 추가
                            current_task_results.extend(new_task_results)
                            
                            return {
                                "messages": scheduled_tasks,
                                "replan_count": state["replan_count"],
                                "task_results": current_task_results,
                                "quick_retriever_message": quick_retriever_message,
                                "key_information": current_key_information
                            }
                    except Exception as e:
                        logging.error(f"Replan execution error: {e}")
                        return state
                
                # 최초 실행 시나리오
                original_query = messages[0].content
                input_query = self.fin_tool.invoke({"query": original_query})
                process_result = self.query_processor_tool.invoke({"query": input_query})
                
                # query_analyzer로 실행 경로 결정
                query_analyzer_result = query_analyzer(process_result.get("input_query"))
                print("query_analyzer_result:", query_analyzer_result)
                
                # 결과를 저장할 변수들 초기화
                task_results = []
                key_information = []
                quick_retriever_message = None
                
                # Quick Retriever Tool 실행이 필요한 경우
                if query_analyzer_result.get("quick_retriever_tool"):
                    quick_retriever_result = self.quick_retriever_tool.invoke(process_result)
                    print("quick_retriever_result:", quick_retriever_result)
                    
                    if quick_retriever_result and quick_retriever_result.get("key_information"):
                        key_info = quick_retriever_result["key_information"][0]
                        key_information = quick_retriever_result["key_information"]
                        
                        # quick_retriever 결과를 task_results에 추가
                        task_results.append({
                            "task": "quick_retriever",
                            "result": quick_retriever_result
                        })
                        
                        quick_retriever_message = FunctionMessage(
                            content=json.dumps(quick_retriever_result),
                            name="quick_retriever_tool",
                            additional_kwargs={
                                "tool": key_info["tool"],
                                "company": key_info["company"],
                                "referenced_content": key_info["referenced_content"],
                                "link": key_info["link"],
                                "idx": 1
                            }
                        )
                        messages.insert(1, quick_retriever_message)
                    
                    # Quick Retriever만 사용하는 경우
                    if not query_analyzer_result.get("plan_and_schedule"):
                        return {
                            "messages": messages,
                            "replan_count": 1,
                            "key_information": key_information,
                            "task_results": task_results,
                            "quick_retriever_message": quick_retriever_message
                        }
                
                # Plan and Schedule 실행이 필요한 경우
                input_query = process_result.get("input_query")
                metadata = process_result.get("metadata", {})
                messages[0].content = input_query
                
                # Report Agent 검사
                if is_valid_query(original_query) and is_in_report_agent(metadata.get("companyName", "")):
                    result_message = self.report_agent_tool.invoke({
                        "query": original_query,
                        "metadata": metadata
                    })
                    return {
                        "messages": [result_message],
                        "replan_count": 1,
                        "report_agent_use": True
                    }
                
                # Plan and Schedule 실행
                try:
                    initial_tasks = list(planner.stream(messages))
                    if initial_tasks:
                        scheduled_tasks, new_task_results = schedule_tasks.invoke({
                            "messages": messages,
                            "tasks": initial_tasks
                        })
                        
                        # plan_and_schedule의 결과를 task_results에 추가
                        task_results.extend(new_task_results)
                        
                        # key_information 업데이트
                        for result in new_task_results:
                            if isinstance(result, dict) and 'result' in result:
                                result_data = result['result']
                                if isinstance(result_data, dict) and 'key_information' in result_data:
                                    key_information.extend(result_data['key_information'])
                        
                        return {
                            "messages": scheduled_tasks,
                            "replan_count": 0,
                            "task_results": task_results,
                            "key_information": key_information,
                            "quick_retriever_message": quick_retriever_message
                        }
                        
                except Exception as e:
                    logging.error(f"Plan execution error: {e}")
                    # 에러가 발생해도 지금까지 수집된 결과 반환
                    return {
                        "messages": messages,
                        "replan_count": 0,
                        "task_results": task_results,
                        "key_information": key_information,
                        "quick_retriever_message": quick_retriever_message
                    }
                    
            except Exception as e:
                logging.error(f"plan_and_schedule 전체 실행 중 예기치 않은 오류: {e}")
                return {
                    "messages": messages,
                    "replan_count": state.get("replan_count", 1),
                    "task_results": state.get("task_results", [])
                }
            
        return plan_and_schedule
