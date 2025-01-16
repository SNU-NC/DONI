from dataclasses import dataclass
from typing import List, Dict, Any, Union
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, FunctionMessage
from plan.reference import TaskResult
from plan.hcx import HyperCLOVA
from ast import literal_eval

@dataclass
class FinalResponse:
    response: str

@dataclass
class Replan:
    feedback: str

@dataclass
class Action:
    type: str
    content: Union[FinalResponse, Replan]

@dataclass
class JoinResult:
    thought: str
    action: Action

def parse_key_information(task_list: List[TaskResult]) -> List[str]:
    """key_information 파싱"""
    key_information = []
    seen = set()
    
    for task in task_list:
        if isinstance(task, dict) and "result" in task:
            result = task["result"]
            if isinstance(result, dict) and "key_information" in result:
                for info in result["key_information"]:
                    key = (info.get('tool'), info.get('filename'), info.get('referenced_content'))
                    if key not in seen:
                        seen.add(key)
                        key_information.append(info)
    return key_information

def parse_message_content(message: BaseMessage) -> str:
    """메시지 content를 타입에 따라 적절히 파싱"""
    content = message.content

    if isinstance(message, HumanMessage):
        return "<USER QUESTION>" + content + "<USER QUESTION>"
    
    if content == 'join':
        return None
    
    if isinstance(message, FunctionMessage):
        if isinstance(content, (float, int)):
            return str(content)
        
        if isinstance(content, str):
            try:
                float(content)
                return content
            except ValueError:
                try:
                    dict_content = literal_eval(content)
                    if isinstance(dict_content, dict) and 'output' in dict_content:
                        return dict_content['output']
                except:
                    return content
        
        if isinstance(content, dict) and 'output' in content:
            return content['output']
            
    return str(content)

def convert_messages_to_dict(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """LangChain 메시지를 HyperCLOVA 형식으로 변환"""
    converted_messages = []
    
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, FunctionMessage):
            role = "assistant"
        else:
            continue
            
        parsed_content = parse_message_content(message)
        if parsed_content is None:
            continue
            
        if isinstance(message, FunctionMessage) and message.name and message.name != 'math':
            parsed_content = f"Function '{message.name}' result: {parsed_content}"
            
        converted_messages.append({
            "role": role,
            "content": parsed_content
        })

    return converted_messages

async def structured_llm_call(messages: List[BaseMessage], is_final: bool = False) -> JoinResult:
    """HyperCLOVA를 사용하여 structured output을 생성하는 함수"""
    
    normal_output_format = {
        "thought": "action type을 선택한 이유와 근거",
        "action": {
            "type": "선택된 행동 유형 ('final_response' 또는 'replan')",
            "content": {
                "final_response": {
                    "response": "[답변]: <핵심 답변>\n[답변 생성 근거]: <답변 도출 과정과 근거>\n"
                },
                "replan": {
                    "feedback": "재계획이 필요한 이유와 제안사항"
                }
            }
        }
    }
    
    final_output_format = {
        "thought": "선택된 행동에 대한 추론 과정을 문자열로 작성",
        "action": {
            "type": "선택된 행동 유형 ('final_response')",
            "content": {
                "final_response": {
                    "response": "[답변]: <핵심 답변>\n[답변 생성 근거]: <답변 도출 과정과 근거>\n[한계 및 개선방안] : <답변의 한계 및 개선방안>"
                }
            }
        }
    }

    system_instructions = """[의사결정 가이드]
<USER QUESTION> 에 대한 답변을 생성해주세요. 

1. thought 작성 시:
   - <USER QUESTION>의 핵심 요구사항 파악
   - 현재 보유한 정보의 충분성 평가
   - 추가 정보 필요 여부 판단
   - 추후 어떤 작업이 필요한지 명시

2. action type 선택 시:
   - replan 선택 조건:
     * message에 답변에 필요한 정보가 부족한 경우
     * 제공받은 정보가 부족해 답변을 할 수 없는 경우 
   - final_response 선택 조건:
     * 필요한 모든 정보가 수집된 경우
     * 데이터의 신뢰성이 확보된 경우"""

    try:
        clova = HyperCLOVA()
        output_format = final_output_format if is_final else normal_output_format
        converted_messages = convert_messages_to_dict(messages)
        
        result = await clova.call_api(
            messages=converted_messages,
            output_format=output_format,
            system_instructions=system_instructions,
            format_class=JoinResult
        )
        return result

    except Exception as e:
        print(f"Error in structured_llm_call: {str(e)}")
        raise

def create_joiner(llm: Any) -> callable:
    """Joiner 함수 생성"""
    
    def joiner(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        task_results = state.get("task_results", [])
        key_info = parse_key_information(task_results)
        report_agent_use = state.get("report_agent_use", False)
        replan_count = state.get("replan_count", 1)
        
        if not messages:
            return {
                "messages": [AIMessage(content="죄송합니다. 계획 수립 중 오류가 발생했습니다.")],
                "replan_count": replan_count + 1,
                "key_information": key_info
            }
            
        try:
            if report_agent_use:
                return {
                    "messages": [AIMessage(content=state["messages"][-1].content)],
                    "key_information": key_info
                }
            
            result = asyncio.run(structured_llm_call(messages, is_final=(replan_count >= 1)))
            
            if not result:
                raise ValueError("structured_llm_call returned None")
                
            thought_message = AIMessage(content=f"Thought: {result.thought}")
            
            if result.action.type == 'replan':
                return {
                    "messages": [
                        thought_message,
                        SystemMessage(content=f"Context from last attempt: {result.action.content.feedback}")
                    ],
                    "key_information": key_info
                }
            else:
                return {
                    "messages": [
                        thought_message,
                        AIMessage(content=result.action.content.response)
                    ],
                    "key_information": key_info
                }
                
        except Exception as e:
            print(f"Error in joiner: {str(e)}")
            return {
                "messages": [AIMessage(content="죄송합니다. 응답을 생성하는 중 오류가 발생했습니다.")],
                "task_results": task_results,
                "key_information": key_info
            }
    
    return joiner 