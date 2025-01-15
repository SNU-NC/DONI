import logging
from typing import List, Union, Dict, Any
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage , FunctionMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import TypedDict
import json
import logging
from ast import literal_eval

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class FinalResponse(BaseModel):
    """The final response/answer."""
    response: str = Field(description="사용자 질문에 대한 상세한 답변")


class JoinerState(TypedDict):
    messages: List[BaseMessage]
    replan_count: int
    force_final_answer: bool

class Replan(BaseModel):
    """Feedback for replanning."""
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decision model for replanning or final response."""
    thought: str = Field(
        description="The chain of thought reasoning for the selected action"

    )
    action: Union[FinalResponse, Replan]


def _parse_joiner_output(decision: JoinOutputs) -> Dict[str, Any]:
    """Parse the joiner output and convert to messages."""
    response = [AIMessage(content=f"Thought: {decision.thought}")]

    if isinstance(decision.action, Replan): 
        return {
            "messages": response + [SystemMessage(content=f"Context from last attempt: {decision.action.feedback}")],
            "key_information": []
        }
    else:
        formatted_response = decision.action.response
        return {
            "messages": response + [AIMessage(content=formatted_response)],
            "key_information": decision.action.key_information if hasattr(decision.action, 'key_information') else []
        }


def select_recent_messages(state) -> dict:
    """Select the most recent messages up to the last human message."""
    messages = state["messages"]
    replan_count = state["replan_count"]
    task_results = state.get("task_results", [])
    
    # output 수집 
    output_list = []
    for message in messages:
        print(f"type of message: {type(message)}")
        print(f"message.content: {message.content}")
        print(f"message.content type: {type(message.content)}")
        print(f"메세지 타입 확인, isHumanMessage: {isinstance(message, HumanMessage)}")
        print(f"메세지 타입 확인, isFunctionMessage: {isinstance(message, FunctionMessage)}")
        try :
            if isinstance(message, HumanMessage):
                output_list.append(message.content)
            elif message.content == 'join':   #메세지 내용이 join인 경우에는 무시 
                print("join 메세지 왔습니다~~~~")
                continue
            elif isinstance(message.content, (float, int)):  # float나 int 타입 체크
                print(f"숫자 타입 메시지 처리: {message.content}")
                output_list.append(str(message.content))
            elif isinstance(message, FunctionMessage):
                print("FunctionMessage에 왔습니다~~~~")
                if isinstance(message.content, str):
                    # 숫자 형태의 문자열인지 확인
                    try:
                        float(message.content)  # 숫자로 변환 시도
                        print(f"숫자 형태의 문자열 처리: {message.content}")
                        output_list.append(message.content)
                    except ValueError:  # 숫자로 변환 실패 시 (일반 문자열)
                        print(f"숫자로 변환 실패 시, 문자열 처리: {message.content}")
                        dict_content = literal_eval(message.content)
                        if isinstance(dict_content['output'], dict):
                            output_list.append(str(dict_content['output']))
                        else : 
                            output_list.append(dict_content['output'])
                        print(f"message.content: {dict_content}")
                        print(f"dict_content['output']: {dict_content['output']}")
                else:
                    print(f"FunctionMessage이지만 문자열이 아닌 타입: {type(message.content)}")


        except Exception as e:
            print(f"예외 발생: {e}, 메시지 타입: {type(message.content)}")
            continue
    print("^^^^^^^^^^^^^^^^^^^^ logging for output_list ^^^^^^^^^^^^^^^^^^^^")
    print(f"output_list: {output_list}")       
    print("^^^^^^^^^^^^^^^^^^^^ logging for output_list  END ^^^^^^^^^^^^^^^^^^^^")
    # key_information 수집
    all_key_information = []
    for result in task_results:
        if isinstance(result, dict):
            if 'result' in result and isinstance(result['result'], dict):
                if 'key_information' in result['result']:
                    all_key_information.extend(result['result']['key_information'])
    print("^^^^^^^^^^^^^^^^^^^^ logging for messages ^^^^^^^^^^^^^^^^^^^^")
    print(f"messages: {messages}")
    print(f"message_type: {type(messages)}")
    print("^^^^^^^^^^^^^^^^^^^^ logging for messages  END ^^^^^^^^^^^^^^^^^^^^")
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    
    return {
        "messages": selected[::-1], 
        "replan_count": replan_count,
        "key_information": all_key_information,
        "output_list": output_list
    }


def check_replan_count(state: dict):
    """Check replan count and decide next step."""
    messages = state["messages"]
    replan_count = state["replan_count"]
    output_list = state["output_list"]
    if replan_count >= 2:
        print(f"check replan count: {replan_count} forcing final answer")
        return {
            "messages": messages,
            "replan_count": replan_count,
            "force_final_answer": True
        }
    print(f"check replan count: {replan_count} continuing")
    return {
        "messages":[SystemMessage(content=msg) for msg in output_list],
        "replan_count": replan_count,
        "force_final_answer": False,
        "output_list": output_list
    }



def create_joiner(llm: BaseChatModel):
    """Create a joiner chain with the given LLM."""
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d")
    
    joiner_prompt = hub.pull("snunc/rag-llm-compiler-joiner-v2").partial(
        examples="", 
        time=current_time,
        format_instructions=""
    )

    final_prompt = hub.pull("snunc/rag-llm-compiler-final-answer").partial(
        examples="",
        time=current_time,
   )
    
    # 함수들을 Runnable로 변환
    select_messages_runnable = RunnableLambda(select_recent_messages)
    check_replan_runnable = RunnableLambda(check_replan_count)
    parse_output_runnable = RunnableLambda(_parse_joiner_output)
    
    # 경로 정의
    normal_path = joiner_prompt | llm.with_structured_output(JoinOutputs)
    replan_fail_path = final_prompt | llm.with_structured_output(JoinOutputs)
    
    def should_use_final_answer(state: Dict[str, Any]) -> bool:
        # true 면 replan 답변을 사용하고, false 면 normal 답변을 사용한다.
        should_use_final_answer = False
        if state["replan_count"] >= 2:
            should_use_final_answer = True
            print("shoud we use final answer?" , " yes")
        print("shoud we use final answer?" , " no")
        return should_use_final_answer
    
    final_answer_path = RunnableBranch(
        (should_use_final_answer, replan_fail_path),
        normal_path
    )
    
    # 체인 구성
    chain = (
        select_messages_runnable
        | check_replan_runnable
        | final_answer_path
        | parse_output_runnable
    )
    
    # 체인을 쓸지 말지 결정
    def conditional_joiner(state: Dict[str, Any]) -> Dict[str, Any]:
        report_agent_use = state.get("report_agent_use", False)
        
        # task_results에서 key_information 수집 및 중복 제거
        all_key_information = []
        seen = set()  # 중복 체크를 위한 set
        
        for task in state.get("task_results", []):
            if isinstance(task, dict) and "result" in task:
                result = task["result"]
                if isinstance(result, dict) and "key_information" in result:
                    for info in result["key_information"]:
                        # tool, filename, referenced_content를 기준으로 중복 체크
                        key = (info.get('tool'), info.get('filename'), info.get('referenced_content'))
                        if key not in seen:
                            seen.add(key)
                            all_key_information.append(info)
        
        if report_agent_use:
            return {
                "messages": [AIMessage(content=state["messages"][-1].content)],
                "key_information": all_key_information
            }
        
        result = chain.invoke(state)
        # key_information 추가
        result["key_information"] = all_key_information
        return result

    
    return conditional_joiner