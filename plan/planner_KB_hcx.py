from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Sequence, Union
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import (
    chain as as_runnable,
)
from plan.hcx import HyperCLOVA
import asyncio
from datetime import datetime
from tools.extractor.query_processor_tool import get_query_processor_tool
from tools.financialTerm.fin_knowledge_tools import get_fin_tool
from tools.planKB.plan_store import PlanStore
from langchain_openai import OpenAIEmbeddings
import logging
from plan.scheduler_hcx import schedule_tasks
from functools import wraps
from plan.models import TaskPlan, PlanResult

logging.basicConfig(level=logging.INFO, format='%(message)s')

@dataclass
class Tool:
    name: str
    description: str



class Planner:
    def __init__(self, llm: BaseChatModel, llm_mini: BaseChatModel, llm_clova: BaseChatModel, tools: Sequence[BaseTool]):
        self.llm = llm
        self.llm_mini = llm_mini
        self.llm_clova = llm_clova
        self.tools = tools
        self.plan_store = PlanStore(OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        ))
        self.fin_tool = get_fin_tool(llm_mini)
        self.query_processor_tool = get_query_processor_tool(llm, llm_clova)
        self.clova = HyperCLOVA()
    def create_plan_and_schedule(self, state: Dict[str, Any]):
        """plan_and_schedule 함수를 생성하여 반환"""
        
        async def structured_llm_call(messages: List[BaseMessage], input_query: str) -> PlanResult:
            """HyperCLOVA를 사용하여 structured output을 생성하는 함수"""
            print("structured_llm_call 시작")
            # 쿼리 추출 및 유사 예제 가져오기
            similar_examples = self.plan_store.get_similar_examples(input_query)
            print("logging for similar_examples")
            print(similar_examples)
            
            # planning_candidates 생성
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
            if not planning_candidates :
                planning_candidates = " "

            # 2. output_format 정의 - scheduler가 기대하는 형식으로 수정
            output_format = {
                "thought": "string",
                "tasks": [
                    {
                        "idx": "number",
                        "tool": "string",
                        "args": {
                            "query": "string",
                            "company": "string",
                            "year": "number"
                        },
                        "dependencies": "number[]"
                    }
                ]
            }

            system_instructions = f"""[계획 수립 가이드]
사용자의 질문을 해결하기 위한 최적의 병렬 실행 계획을 수립해주세요.

사용 가능한 도구 목록:
{', '.join([f"{tool.name}: {tool.description}" for tool in self.tools])}
{len(self.tools)+1}. join(): 이전 작업들의 결과를 수집하고 결합하는 도구

### 도구 사용 우선순위
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

### 연도 관련 주의사항
- 사용자가 연도를 지정하지 않은 경우, 기본값은 2023년입니다 (예: 최근 2개년 = 2023년, 2022년)
- 단, 애널리스트 보고서 검색 시에는 기본값을 2024년으로 설정하세요

### 계획 작성 규칙
1. 작업 형식
   - 각 작업은 반드시 위 도구 목록 중 하나여야 합니다
   - Python 규칙을 따라 작업을 작성하세요 (예: tool_name(arg_name=value))
   - 각 작업은 고유한 ID를 가져야 하며, 순차적으로 증가해야 합니다

2. 작업 입력값
   - 상수 또는 이전 작업의 출력을 사용할 수 있습니다
   - 이전 작업의 출력을 참조할 때는 $id 형식을 사용하세요 (id는 참조할 작업의 ID)

3. join() 사용
   - join()은 항상 계획의 마지막 작업이어야 합니다
   - 다음 두 경우에 join()을 호출합니다:
     a) 작업들의 출력을 모아서 최종 응답을 생성할 수 있는 경우
     b) 계획 실행 전에 답변을 결정할 수 없는 경우

4. 병렬 실행
   - 가능한 한 작업들이 병렬로 실행될 수 있도록 계획을 수립하세요
   - 작업 간의 의존성을 명확히 표시하세요
### 반드시 지키시오 
join() 사용
   - join()은 항상 계획의 마지막 작업이어야 합니다
   - join은 모든 작업들에 대해 dependencies를 가집니다. 
### 참고할 만한 유사 논리적인 계획 예시
{planning_candidates}"""

            try:
                print("들어갈 메세지 파싱 시작")
                converted_messages = [
                    {
                        "role": "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else "system",
                        "content": msg.content
                    }
                    for msg in messages
                ]
                print("이제 클로바 부를게요~")
                # 4. API 호출
                result = await self.clova.call_api(
                    messages=converted_messages,
                    output_format=output_format,
                    system_instructions=system_instructions,
                    format_class=PlanResult,
                    temperature=0.7
                )
        
                print("structured_llm_call result:", result)
                return result


            except Exception as e:
                print(f"Error in structured_llm_call: {str(e)}")
                raise

        @as_runnable
        def plan_and_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
            """플래너 메인 함수"""
            messages = state["messages"]
            print("Plan*Scheduler 시작할 때의 replan Count를 체크하겠습니다.", state.get("replan_count", 404))
            
            try:
                # 원본 쿼리 저장
                original_query = messages[0].content
                print("original_query:", original_query)
                
                # fin_tool을 사용하여 금융 지식 주입
                input_query = original_query
                if "replan_count" not in state:
                    input_query = self.fin_tool.invoke({"query": original_query})
                    process_result = self.query_processor_tool.invoke({"query": input_query})
                    print("process_result:", process_result)
                    input_query = process_result["input_query"]
                    metadata = process_result["metadata"]
                
                # 보강된 쿼리로 메시지 업데이트
                messages[0].content = input_query
                logging.info(f"원본 쿼리: {original_query}")
                logging.info(f"FinTool 사용 후의 쿼리: {input_query}")
                
                try:
                    # HyperCLOVA를 사용하여 계획 생성
                    result = asyncio.run(structured_llm_call(messages, input_query))
                    if not result:
                        raise ValueError("structured_llm_call returned None")
                                    # 디버깅을 위한 로깅 추가
                    print("\n=== HyperCLOVA 생성 계획 디버깅 ===")
                    print(f"생각 과정: {result.thought}")
                    print("\n계획된 작업들:")
                    for i, task in enumerate(result.tasks):
                        print(f"\n작업 {i+1}:")
                        print(f"도구: {task.tool}")
                        print(f"매개변수: {task.args}")
                    print("================================\n")

                except Exception as e:
                    logging.error(f"초기 태스크 생성 중 오류 발생: {e}")
                    return {
                        "messages": [],
                        "replan_count": state.get("replan_count", 1),
                        "task_results": state.get("task_results", [])
                    }

                # 스케줄링 작업
                try:
                    if result.tasks:
                        logging.info(f"스케줄링할 태스크 수: {len(result.tasks)}")
                        print("###########스케줄링할 태스크 result.tasks: ", result.tasks)
                        print("###########스케줄링할 태스크 수 just result: ", result)
                        scheduled_tasks, new_task_results = schedule_tasks.invoke({
                            "messages": messages,
                            "tasks": result.tasks
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

                # replan count 관리
                if "replan_count" not in state:
                    state["replan_count"] = 0
                    print("replan_count 초기화", state["replan_count"])
                else:
                    state["replan_count"] = state["replan_count"] + 1

                return {
                    "messages": [HumanMessage(content=f" {original_query}")] + scheduled_tasks,
                    "replan_count": state["replan_count"],
                    "task_results": current_task_results,
                    "metadata": metadata if "metadata" in locals() else None
                }
                    
            except Exception as e:
                logging.error(f"plan_and_schedule 전체 실행 중 예기치 않은 오류: {e}")
                return {
                    "messages": [],
                    "replan_count": state.get("replan_count", 1),
                    "task_results": state.get("task_results", [])
                }
        
        return plan_and_schedule
