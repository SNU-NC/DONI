from datetime import datetime
import re
import time
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Union, Iterable
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.runnables import (
    chain as as_runnable,
)
from typing_extensions import TypedDict
import logging
from langchain.tools import BaseTool

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results

class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Dict[str, Any]]

class ToolRegistry:
    """도구 레지스트리 클래스"""
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, name: str, tool: BaseTool):
        cls._tools[name] = tool
    
    @classmethod
    def get_tool(cls, name: str) -> BaseTool:
        return cls._tools.get(name)

def _execute_task(task, observations, config):
    tool_to_use = task.tool
    logger.debug(f"\n{'='*50}")
    logger.debug(f"Executing task {task.idx} with tool: {type(tool_to_use)}")
    
    print("tool_to_use 를 확인해보겠습니다. ")
    print(tool_to_use)
    
    # 문자열인 경우 tools 모듈에서 해당하는 도구 클래스를 찾아 인스턴스화
    if isinstance(tool_to_use, str):
        print("tool_to_use 가 문자열인 경우")
        print(tool_to_use)
        
        # 도구 임포트
        from tools.webSearch.webSearch_tool import WebSearchTools
        from tools.retrieve.analystReport.report_RAG_Tool import ReportRAGTool
        from tools.retrieve.financialReport.report_statement_tool import CombinedFinancialReportSearchTool
        from tools.math.math_tools import get_math_tool
        from tools.financialTerm.fin_knowledge_tools import get_fin_tool
        from tools.marketData.market_agent import MarketDataTool
        from tools.analyze.stockprice.SameSectorCompareTool import SameSectorAnalyzerTool
        from tools.analyze.stockprice.StockAnalyzerTool import StockAnalyzerTool
        from tools.analyze.stockprice.CombinedAnalysisTool import CombinedAnalysisTool
        
        # LLM 초기화 (필요한 경우)
        llm_4o = ChatOpenAI(model_name="gpt-4o", temperature=0.8)
        llm_mini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)
        
        # 도구 매핑
        tool_mapping = {
            'web_search': lambda: WebSearchTools(llm_mini),
            'math': lambda: get_math_tool(llm_4o),
            'report_rag_search': ReportRAGTool,
            'StockAnalyzerTool': StockAnalyzerTool,
            'CombinedAnalysisTool': CombinedAnalysisTool,
            'SameSectorAnalyzer': SameSectorAnalyzerTool,
            'combined_financial_report_search': CombinedFinancialReportSearchTool,
            'market_data_tool': MarketDataTool,
        }
        
        print("tool_mapping 를 확인해보겠습니다. ")
        print(tool_mapping)
        
        if tool_to_use in tool_mapping:
            tool_class = tool_mapping[tool_to_use]
            # 함수인 경우 호출, 클래스인 경우 인스턴스화
            tool_to_use = tool_class() if callable(tool_class) else tool_class
        else:
            return f"ERROR: Tool {task.tool} not found"
    
    args = task.args
    print("(resolved 전)*args check *************")
    print(args)
    
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: val if key == 'problem' else _resolve_arg(val, observations)
                for key, val in args.items()
            }
        else:
            resolved_args = args
            
        print("tool에 진짜 들어가는 형태의 args")
        print(resolved_args)
        
        # 도구 실행
        result = tool_to_use.invoke(resolved_args, config)
        return result
        
    except Exception as e:
        logger.error(f"Task execution failed: {str(e)}", exc_info=True)
        return f"ERROR: {str(e)}"

def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    """인자 해석 및 변환 함수"""
    ID_PATTERN = r"\$\{?(\d+)\}?"
    #ID_PATTERN = r"\$(\d+)"
    print("*"*30)
    print("*Whatis my arg ? *************")
    print(arg)
    print("*Observations check *************")
    print(observations)

    def extract_value(value: Any) -> str:
        """관찰값에서 text 필드 추출"""
        if value is None:
            return ""
            
        # JSON 형태 처리 
        if isinstance(value, dict):
            if 'output' in value:
                if isinstance(value['output'], dict):
                    # text 필드 우선 사용
                    if 'text' in value['output']:
                        return value['output']['text']
                return str(value['output'])
            return str(value)
            
        return str(value)

    if isinstance(arg, str):
        # context 파라미터 처리 ([$1, $2] 형태)
            
        matches = re.findall(ID_PATTERN, arg)
        if matches:
            results = []
            for idx in matches:
                obs = observations.get(int(idx))
                if obs:
                    results = extract_value(obs)
            return results
    
        # problem 필드는 그대로 유지 (치환 없이)
        if isinstance(arg, str) and not arg.startswith("["):
            return arg
            
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
        
    return arg
# Math 문제 생기기 이전 _resolve_arg 함수 
# def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
#     ID_PATTERN = r"\$(\d+)"
    
#     if isinstance(arg, str):
#         # context 파라미터 처리를 위한 특수 케이스
#         if arg.startswith("[") and arg.endswith("]"):
#             # $1, $2 형태의 참조들을 찾아서 리스트로 변환
#             matches = re.findall(ID_PATTERN, arg)
#             if matches:
#                 return [observations.get(int(idx)) for idx in matches]
        
#         # 일반적인 문자열 치환
#         return re.sub(ID_PATTERN, lambda m: str(observations.get(int(m.group(1)), m.group(0))), arg)
#     elif isinstance(arg, list):
#         return [_resolve_arg(a, observations) for a in arg]
#     return arg

# def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
#     # $1 or ${1} -> 1
#     ID_PATTERN = r"\$\{?(\d+)\}?"

#     def replace_match(match):
#         # If the string is ${123}, match.group(0) is ${123}, and match.group(1) is 123.

#         # Return the match group, in this case the index, from the string. This is the index
#         # number we get back.
#         idx = int(match.group(1))

#     # For dependencies on other tasks
#     if isinstance(arg, str):
#         return re.sub(ID_PATTERN, replace_match, arg)
#     elif isinstance(arg, list):
#         return [_resolve_arg(a, observations) for a in arg]
#     else:
#         return str(arg)



@as_runnable
def schedule_task(task_inputs, config):
    task = task_inputs["task"]
    observations = task_inputs["observations"]
    try:
        logger.debug(f"Scheduling task: {task}")
        logger.debug(f"Task tool type: {type(task.tool)}")
        observation = _execute_task(task, observations, config)
        logger.debug(f"Task result type: {type(observation)}")
    except Exception as e:
        logger.error("Task execution failed", exc_info=True)
        observation = str(e)  # 예외를 문자열로 변환
    
    
    try:
        logger.debug(f"Attempting to serialize observation for task {task.idx}")
        observations[task.idx] = observation
    except Exception as e:
        logger.error(f"Failed to serialize observation", exc_info=True)
        observations[task.idx] = str(observation)

def schedule_pending_task(task, observations, retry_after: float = 0.2):
    while True:
        deps = task.dependencies  # 직접 접근
        if deps and (any([dep not in observations for dep in deps])):
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break

@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    tasks = scheduler_input["tasks"]
    logger.debug(f"Received tasks: {len(tasks)} tasks")
    
    # tasks가 비어있는 경우 경고 로그를 출력
    if not tasks:
        print("스케줄링할 태스크가 없음")
        print(f"scheduler_input : {scheduler_input}")
        logger.warning("스케줄링할 태스크가 없음")
        return []
        
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    futures = []
    retry_after = 0.25

    with ThreadPoolExecutor() as executor:
        for task in tasks:
            try:
                # TaskPlan 객체의 속성에 직접 접근
                logger.debug(f"Processing task {task.idx}")  # .get() 대신 직접 접근
                logger.debug(f"Task tool type: {type(task.tool)}")
                
                deps = task.dependencies  # .get() 대신 직접 접근
                task_names[task.idx] = (
                    task.tool if isinstance(task.tool, str) else task.tool.name
                )
                args_for_tasks[task.idx] = task.args
                
                if deps and (any([dep not in observations for dep in deps])):
                    logger.debug(f"Task {task.idx} has pending dependencies: {deps}")
                    futures.append(
                        executor.submit(
                            schedule_pending_task, task, observations, retry_after
                        )
                    )
                else:
                    logger.debug(f"Executing task {task.idx} immediately")
                    schedule_task.invoke(dict(task=task, observations=observations))
            except Exception as e:
                logger.error(f"Failed to process task {task.idx}", exc_info=True)
                
        wait(futures)
    print("observations")
    try:
        logger.debug("Creating tool messages")
        new_observations = {
            k: (task_names[k], args_for_tasks[k], observations[k])
            for k in sorted(observations.keys() - originals)
        }
        # 태스크 결과 저장
        task_results = []
        for k, (name, task_args, obs) in new_observations.items():
            task_results.append({
                "tool_name": name,
                "task_id": str(k),
                "timestamp": datetime.now().isoformat(),
                "result": obs,
                "metadata": {
                    "arguments": task_args,
                    # 도구별 특수 메타데이터 추가 가능
                }
            })
        
        tool_messages = [
            FunctionMessage(
                name=name,
                content=str(obs),
                additional_kwargs={"idx": k, "args": task_args},
                tool_call_id=k,
            )
            for k, (name, task_args, obs) in new_observations.items()
        ]
        logger.debug(f"Created {len(tool_messages)} tool messages")
        return tool_messages , task_results
    except Exception as e:
        logger.error("Failed to create tool messages", exc_info=True)
  