from ast import  literal_eval
import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Union, Callable ,List
from plan.reference import TaskResult
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, FunctionMessage
import traceback
from dataclasses import dataclass

class FinalResponse(BaseModel):
    response: str = Field(description="사용자 질문에 대한 상세한 답변")

class Replan(BaseModel):
    feedback: str = Field(description="이전 시도의 분석 및 수정 필요 사항에 대한 제안")

class JoinOutputs(BaseModel):
    thought: str = Field(description="선택된 행동에 대한 추론 과정")
    action: Union[FinalResponse, Replan]

@dataclass
class Action:
    type: str
    response: str = None
    feedback: str = None

@dataclass
class JoinResult:
    thought: str
    action: Action


def parse_clova_response(result: Dict[str, Any]) -> JoinResult:
    """HyperCLOVA API 응답을 JoinResult 객체로 파싱하는 함수"""
    if not (result and 'result' in result and 'message' in result['result']):
        raise ValueError("Invalid API response format")
    
    try:
        # content 문자열을 파이썬 객체로 파싱
        content_str = result['result']['message']['content']
        content_obj = json.loads(content_str)
        
        # action 객체 생성
        if content_obj['action']['type'] == 'final_response':
            action = Action(
                type='final_response',
                response=content_obj['action']['content']['final_response']['response']
            )
        else:
            action = Action(
                type='replan',
                feedback=content_obj['action']['content']['replan']['feedback']
            )
        
        # 최종 결과 객체 생성
        return JoinResult(
            thought=content_obj['thought'],
            action=action
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse API response content: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required field in API response: {e}")    
def parse_key_information(task_list: List[TaskResult]) -> List[str]:
    """key_information 파싱"""
    key_information = []
    seen = set()
    print("초기 task list 받았습니다 ")
    for task in task_list :
        if isinstance(task, dict) and "result" in task:
            result = task["result"]
            if isinstance(result, dict) and "key_information" in result:
                for info in result["key_information"]:
                    # tool, filename, referenced_content를 기준으로 중복 체크
                    key = (info.get('tool'), info.get('filename'), info.get('referenced_content'))
                    if key not in seen:
                        seen.add(key)
                        key_information.append(info)
    print("key information으로 변환 했습니다", key_information)
    return key_information


def parse_message_content(message: BaseMessage) -> str:
    """메시지 content를 타입에 따라 적절히 파싱하여 반환 (메세지의 output만 받도록)"""
    content = message.content

    # HumanMessage인 경우 content 그대로 반환
    if isinstance(message, HumanMessage):
        return "<USER QUESTION>" +content +"<USER QUESTION>"
    
    # content가 'join'인 경우 None 반환 (이후 처리에서 제외)
    if content == 'join':
        return None
    
    # FunctionMessage 처리
    if isinstance(message, FunctionMessage):
        if isinstance(content, (float, int)):
            return str(content)
        
        if isinstance(content, str):
            try:
                # 숫자 형태의 문자열인지 확인
                float(content)
                return content
            except ValueError:
                # 딕셔너리 형태의 문자열 파싱 시도
                try:
                    dict_content = literal_eval(content)
                    if isinstance(dict_content, dict) and 'output' in dict_content:
                        return dict_content['output']
                except:
                    return content
        
        if isinstance(content, dict) and 'output' in content:
            return content['output']
            
    # 기본적으로는 문자열로 변환하여 반환
    return str(content)


def convert_messages_to_dict(messages: list[BaseMessage]) -> list[dict]:
    """LangChain 메시지 객체를 HyperCLOVA 형식으로 변환"""
    """ 각 수행 결과의 output 만 참고하도록 수정 """
    converted_messages = []
    for message in messages:
        # 기본 role 설정
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
            
        # content 파싱
        parsed_content = parse_message_content(message)
        if parsed_content is None:  # 'join' 메시지 등 제외할 내용
            continue
            
        # FunctionMessage의 경우 함수 실행 결과 정보 추가
        # Function 이름 추가는 math 함수가 아닌 경우에만
        if isinstance(message, FunctionMessage) and message.name and message.name != 'math':
            parsed_content = f"Function '{message.name}' result: {parsed_content}"
            
        converted_messages.append({
            "role": role,
            "content": parsed_content
        })

    print("converted_messages:", converted_messages)
    return converted_messages

async def call_clova_api(messages: list, structured_output_format: Dict[str, Any],) -> Dict[str, Any]:
    """HyperCLOVA API를 호출하여 structured output을 반환하는 함수"""
    
    print("\n=== CLOVA API 호출 시작 ===")
    print(f"입력 메시지: {messages}")
    print(f"출력 형식: {json.dumps(structured_output_format, indent=2, ensure_ascii=False)}")
    
    # 시스템 메시지에 structured output 형식 추가
    system_prompt = f"""다음 JSON 형식으로만 응답해주세요:
{json.dumps(structured_output_format, indent=2, ensure_ascii=False)}

[의사결정 가이드]
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
     * 데이터의 신뢰성이 확보된 경우

주의사항:
1. 반드시 위 JSON 형식을 따라주세요
2. JSON 형식 외의 다른 텍스트는 포함하지 마세요
3. 모든 필드를 반드시 포함해주세요
4. <USER QUESTION>를 해결하기 위한 답변은 반드시 제공받은 정보를 기반으로 생성해야 하며, 필요한 정보가 없다면 정보가 없다고 명시해주세요.
"""

    # 메시지 변환 및 시스템 프롬프트 추가
    converted_messages = convert_messages_to_dict(messages)
    print(f"\n변환된 메시지: {converted_messages}")
    
    converted_messages.insert(0, {
        "role": "system",
        "content": system_prompt
    })
    
    print(f"시스템 메세지가 추가된 변환된 메세지: {converted_messages}")    
    url = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"
    
    headers = {
        "X-NCP-CLOVASTUDIO-API-KEY": os.getenv("CLOVA_API_KEY"),
        "X-NCP-APIGW-API-KEY": os.getenv("CLOVA_API_GATEWAY_KEY"),
        "Content-Type": "application/json"
    }
    
    request_data = {
        "messages": converted_messages,
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 1024,
        "temperature": 0.1,
        "repeatPenalty": 5.0,
        "stopBefore": [],
        "includeAiFilters": True
    }
    
    try:
        print("\n=== API 요청 시작 ===")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data, headers=headers) as response:
                print(f"API 응답 상태: {response.status}")
                result = await response.json()
                print(f"받은 라인 원본 result: {result}")
                print(f"받은 라인: {json.dumps(result, indent=2, ensure_ascii=False)}")
                temp_message = result["result"]["message"]
                print(f"result[message]:{temp_message}")

                return parse_clova_response(result)

    except Exception as e:
        print(f"API 호출 중 오류 발생: {str(e)}")
        raise

async def structured_llm_call(messages: list , is_final: bool = False) -> JoinOutputs:
    """HyperCLOVA를 사용하여 structured output을 생성하는 함수"""
    
    print("\n=== structured_llm_call 시작 ===")
    print(f"입력 메시지: {messages}")
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
            "type": "선택된 행동 유형 ('final_response' )",
            "content": {
                "final_response": {
                    "response": "[답변]: <핵심 답변>\n[답변 생성 근거]: <답변 도출 과정과 근거>\n[한계 및 개선방안] : <답변의 한계 및 개선방안>"
                }
            }
        }
    }
    output_format = normal_output_format if not is_final else final_output_format


    try:
        result = await call_clova_api(messages, output_format)
        print(f"\nAPI 호출 결과: {result}")
        return result

        
    except Exception as e:
        print(f"Error in structured_llm_call: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise

def create_joiner(llm: Any) -> Callable:
    """Joiner 함수 생성"""
    
    def joiner(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        task_results = state.get("task_results", [])
        key_info = parse_key_information(task_results)
        report_agent_use = state.get("report_agent_use", False)
        try :
            replan_count = state.get("replan_count" , 1)
            print("******join 시작부분 replan_count 로깅 ", replan_count)
        except :
            print("replan_count 없음")
        # 1. messages가 비어있는 경우 처리
        if not messages:
            return {
                "messages": [AIMessage(content="죄송합니다. 계획 수립 중 오류가 발생했습니다.")],
                "replan_count": replan_count + 1,
                "key_information": key_info
            }
            
        try:

            ##  보고서 작성 시  
            if report_agent_use :
                return {
                    "messages": [AIMessage(content=state["messages"][-1].content)],
                    "key_information": key_info
                }
            
            # 보고서 작성 X 시
            if replan_count >=1 : 
                print("replan_count >=1 이므로 final_output_format 사용")
                result = asyncio.run(structured_llm_call(messages, is_final=True))
            else :
                print("replan_count <1 이므로 normal_output_format 사용")
                result = asyncio.run(structured_llm_call(messages))
            if not result:  # None 체크
                raise ValueError("structured_llm_call returned None")
                
            print("joiner_test_result:", result)
            
            # 4. 결과 처리 및 변환 개선
            thought_message = AIMessage(content=f"Thought: {result.thought if hasattr(result, 'thought') else '추가 분석이 필요합니다.'}")
            
            if result.action.type == 'replan':
                return {
                    "messages": [
                        thought_message,
                        SystemMessage(content=f"Context from last attempt: {result.action.feedback}")
                    ],
                    "key_information": key_info
                }
            else:
                return {
                    "messages": [
                        thought_message,
                        AIMessage(content=result.action.response if hasattr(result.action, 'response') else '응답을 생성할 수 없습니다.')
                    ],
                    "key_information": key_info
                }
                
        except Exception as e:
            print(f"Error in joiner: {str(e)}")
            error_message = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."

            return {
                "messages": [AIMessage(content=error_message)],
                "task_results": task_results,
                "key_information": key_info
            }
    
    return joiner

# # 테스트용 코드는 if __name__ == "__main__": 블록으로 이동
# if __name__ == "__main__":
#     async def main():
#         messages = [
#             {
#                 "role": "user",
#                 "content": "삼성전자의 2021년부터 2023년까지 영업이익 평균을 계산해줘"
#             }
#         ]
        
#         try:
#             result = await structured_llm_call(messages)
#             print("Structured Output:", result.json(indent=2, ensure_ascii=False))
#         except Exception as e:
#             print(f"Error in main: {str(e)}")

#     asyncio.run(main()) 