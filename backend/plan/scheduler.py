from datetime import datetime
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Union, Iterable
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.runnables import (
    chain as as_runnable,
)
from typing_extensions import TypedDict
import logging
import requests
import json

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

def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    logger.debug(f"\n{'='*50}")
    logger.debug(f"Executing task {task.get('idx')} with tool: {type(tool_to_use)}")
    logger.debug(f"Task details: {task}")
    
    if isinstance(tool_to_use, str):
        logger.debug(f"Tool is string: {tool_to_use}")
        return tool_to_use
        
    args = task["args"]
    print("(resolved 전)*args check *************")
    print(args)
    try:
        if isinstance(args, str):
            logger.debug(f"\nResolving string args: {args}")
            resolved_args = _resolve_arg(args, observations)
            logger.debug(f"Resolved to: {resolved_args}")

        elif isinstance(args, dict):
            print(f"\nResolving dict args: {args}")
            resolved_args = {
                key: val if key == 'problem' else _resolve_arg(val, observations)
                for key, val in args.items()
            }
            
        else:
            logger.debug(f"\nUsing raw args: {args}")
            resolved_args = args
        print("tool에 진짜 들어가는 형태의 args")
        print(resolved_args)        
        print("observations", observations)
 
            
    except Exception as e:
        logger.error(f"Args resolution failed for tool {tool_to_use.__class__.__name__}", exc_info=True)
        return f"ERROR(Failed to resolve args for {tool_to_use.__class__.__name__}. Error: {repr(e)})"


    print("tool에 진짜 들어가는 형태의 args")
    print(resolved_args)
    try:
        logger.debug(f"\nStarting task execution: {task.get('idx')}")
        logger.debug(f"Invoking tool {tool_to_use.__class__.__name__} with args: {resolved_args}")
        result = tool_to_use.invoke(resolved_args, config)
        logger.debug(f"\nTool execution result: {result}")
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
        logger.debug(f"Task tool type: {type(task.get('tool'))}")
        observation = _execute_task(task, observations, config)
        logger.debug(f"Task result type: {type(observation)}")
    except Exception as e:
        logger.error("Task execution failed", exc_info=True)
        import traceback
        observation = traceback.format_exception()
    
    try:
        logger.debug(f"Attempting to serialize observation for task {task.get('idx')}")
        observations[task["idx"]] = observation
    except Exception as e:
        logger.error(f"Failed to serialize observation", exc_info=True)
        observations[task["idx"]] = str(observation)

def schedule_pending_task(task, observations, retry_after: float = 0.2):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break

@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    tasks = scheduler_input["tasks"]
    logger.debug(f"Received tasks: {tasks}")
    logger.debug(f"Received tasks: {len(tasks)} tasks")  

    args_for_tasks = {}
    messages = scheduler_input["messages"]
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    futures = []
    retry_after = 0.25

    def serialize_tool(tool):
        """도구 객체를 직렬화 가능한 형태로 변환"""
        if isinstance(tool, str):
            return tool
        try:
            return {
                "name": tool.name if hasattr(tool, "name") else tool.__class__.__name__,
                "type": tool.__class__.__name__
            }
        except:
            return str(tool)

    def serialize_result(result):
        """결과를 직렬화 가능한 형태로 변환"""
        if isinstance(result, (str, int, float, bool, type(None))):
            return result
        if isinstance(result, dict):
            return {k: serialize_result(v) for k, v in result.items()}
        if isinstance(result, (list, tuple)):
            return [serialize_result(item) for item in result]
        return str(result)

    with ThreadPoolExecutor() as executor:
        for task in tasks:
            try:
                logger.debug(f"Processing task {task.get('idx')}")
                logger.debug(f"Task tool type: {type(task.get('tool'))}")
                
                deps = task["dependencies"]
                tool = task["tool"]
                serialized_tool = serialize_tool(tool)
                
                task_names[task["idx"]] = (
                    tool if isinstance(tool, str) else getattr(tool, "name", tool.__class__.__name__)
                )
                args_for_tasks[task["idx"]] = task["args"]
                
                # 각 태스크 실행 직후 결과 전송
                execution_data = {
                    "type": "execution",
                    "timestamp": datetime.now().isoformat(),
                    "status": "running",
                    "task_id": task["idx"],
                    "task_name": task_names[task["idx"]],
                    "tool": serialized_tool,
                    "args": serialize_result(args_for_tasks[task["idx"]]),
                    "debug_info": {
                        "task_type": str(type(tool)),
                        "dependencies": deps
                    }
                }
                
                try:
                    response = requests.post(
                        'http://localhost:8000/api/task-progress',
                        json=execution_data,
                        timeout=0.5
                    )
                    logger.debug(f"Task progress update response: {response.status_code}")
                except Exception as e:
                    logger.error(f"Task progress update failed: {e}")
                
                if deps and (any([dep not in observations for dep in deps])):
                    logger.debug(f"Task {task['idx']} has pending dependencies: {deps}")
                    futures.append(
                        executor.submit(
                            schedule_pending_task, task, observations, retry_after
                        )
                    )
                else:
                    logger.debug(f"Executing task {task['idx']} immediately")
                    result = schedule_task.invoke(dict(task=task, observations=observations))
                    
                    # 태스크 완료 후 결과 전송
                    execution_data.update({
                        "status": "completed",
                        "result": serialize_result(observations.get(task["idx"])),
                        "debug_info": {
                            **execution_data["debug_info"],
                            "result_type": str(type(result)) if result else None,
                            "observation_keys": list(observations.keys())
                        }
                    })
                    
                    try:
                        response = requests.post(
                            'http://localhost:8000/api/task-progress',
                            json=execution_data,
                            timeout=0.5
                        )
                        logger.debug(f"Task completion update response: {response.status_code}")
                    except Exception as e:
                        logger.error(f"Task completion update failed: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to process task {task.get('idx')}", exc_info=True)
                # 에러 상황도 전송
                error_data = {
                    **execution_data,
                    "status": "error",
                    "error": str(e),
                    "error_type": e.__class__.__name__
                }
                try:
                    requests.post(
                        'http://localhost:8000/api/task-progress',
                        json=error_data,
                        timeout=0.5
                    )
                except:
                    pass
                
        wait(futures)
    
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
                "result": serialize_result(obs),
                "metadata": {
                    "arguments": serialize_result(task_args),
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
        
        # 최종 결과 전송
        final_data = {
            "type": "execution_summary",
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "task_count": len(task_results),
            "results": task_results
        }
        
        try:
            requests.post(
                'http://localhost:8000/api/task-progress',
                json=final_data,
                timeout=0.5
            )
        except Exception as e:
            logger.error(f"Final results update failed: {e}")
            
        return tool_messages, task_results
        
    except Exception as e:
        logger.error("Failed to create tool messages", exc_info=True)
        return [], []  # 에러 발생 시 빈 리스트 반환
  