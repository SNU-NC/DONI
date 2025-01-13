import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Union, Callable
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

def convert_messages_to_dict(messages: list[BaseMessage]) -> list[dict]:
    """LangChain 메시지 객체를 HyperCLOVA 형식으로 변환"""
    converted_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "assistant"
            content = message.content
        elif isinstance(message, SystemMessage):
            role = "system"
            content = message.content
        elif isinstance(message, FunctionMessage):
            role = "assistant"  # FunctionMessage는 assistant로 처리
            # FunctionMessage의 content와 추가 정보를 포함
            if isinstance(message.content, dict):
                content = json.dumps(message.content, ensure_ascii=False)
            else:
                content = str(message.content)
            
            # 함수 실행 결과에 대한 추가 정보 포함
            if message.name:
                content = f"Function '{message.name}' result: {content}"
            
        else:
            print(f"Warning: Unhandled message type: {type(message)}")
            continue
            
        converted_messages.append({
            "role": role,
            "content": content
        })
    
    return converted_messages

async def call_clova_api(messages: list, structured_output_format: Dict[str, Any]) -> Dict[str, Any]:
    """HyperCLOVA API를 호출하여 structured output을 반환하는 함수"""
    
    print("\n=== CLOVA API 호출 시작 ===")
    print(f"입력 메시지: {messages}")
    print(f"출력 형식: {json.dumps(structured_output_format, indent=2, ensure_ascii=False)}")
    
    # 시스템 메시지에 structured output 형식 추가
    system_prompt = f"""다음 JSON 형식으로만 응답해주세요:
{json.dumps(structured_output_format, indent=2, ensure_ascii=False)}

주의사항:
1. 반드시 위 JSON 형식을 따라주세요
2. JSON 형식 외의 다른 텍스트는 포함하지 마세요
3. 모든 필드를 반드시 포함해주세요"""

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
                print(f"content:{temp_message}")

                if result and 'result' in result and 'message' in result['result']:
                    # content 문자열을 파이썬 객체로 파싱
                    content_str = result['result']['message']['content']
                    content_obj = json.loads(content_str)
                    
                    # LangGraph Join 노드 형식에 맞게 변환
                    @dataclass
                    class Action:
                        type: str
                        response: str = None
                        feedback: str = None
                    
                    @dataclass
                    class JoinResult:
                        thought: str
                        action: Action
                    
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
                else:
                    raise ValueError("Invalid API response format")

    except Exception as e:
        print(f"API 호출 중 오류 발생: {str(e)}")
        raise

async def structured_llm_call(messages: list) -> JoinOutputs:
    """HyperCLOVA를 사용하여 structured output을 생성하는 함수"""
    
    print("\n=== structured_llm_call 시작 ===")
    print(f"입력 메시지: {messages}")
    
    output_format = {
        "thought": "선택된 행동에 대한 추론 과정을 문자열로 작성",
        "action": {
            "type": "선택된 행동 유형 ('final_response' 또는 'replan')",
            "content": {
                "final_response": {
                    "response": "[요약]: <핵심 답변>\n[답변 생성 근거]: <근거>\n[참고 데이터]: <출처>"
                },
                "replan": {
                    "feedback": "재계획이 필요한 이유와 제안사항"
                }
            }
        }
    }
    
    try:
        result = await call_clova_api(messages, output_format)
        print(f"\nAPI 호출 결과: {result}")
        return result
        ### 원래 이런식으로 형식 변환을 진행했는데 걍 쓰면 해결이 되는 문제였다고 한다 
        # if not result:
        #     print("API 결과가 None입니다")
        #     raise ValueError("API returned None")
            
        # # 결과를 JoinOutputs 형식으로 변환
        # if result['action']['type'] == 'final_response':
        #     action = FinalResponse(response=result['action']['content']['final_response']['response'])
        # else:
        #     action = Replan(feedback=result['action']['content']['replan']['feedback'])
            
        # output = JoinOutputs(
        #     thought=result['thought'],
        #     action=action
        # )
        # print(f"\n최종 출력: {output}")
        # return output
        
    except Exception as e:
        print(f"Error in structured_llm_call: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise

def create_joiner(llm: Any) -> Callable:
    """Joiner 함수 생성"""
    
    def joiner(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        replan_count = state.get("replan_count", 0)
        task_results = state.get("task_results", [])
        
        # 1. messages가 비어있는 경우 처리
        if not messages:
            return {
                "messages": [AIMessage(content="죄송합니다. 계획 수립 중 오류가 발생했습니다.")],
                "replan_count": replan_count + 1,
                "task_results": task_results
            }
            
        try:
            # task_results 처리 -> 이거 없앨 수도 일단 내비둠 
            task_content = ""
            if task_results:
                task_content = "\n".join([
                    f"Task Result {i+1}: {result.result}" 
                    for i, result in enumerate(task_results)
                    if hasattr(result, 'result')  # result 속성 존재 확인
                ])
                if task_content:
                    print("task_content:", task_content)
                    messages.append(SystemMessage(content=f"Previous task results:\n{task_content}"))
            
            
            # 3. structured_llm_call 결과 검증
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
                    "replan_count": replan_count + 1,
                    "task_results": task_results
                }
            else:
                return {
                    "messages": [
                        thought_message,
                        AIMessage(content=result.action.response if hasattr(result.action, 'response') else '응답을 생성할 수 없습니다.')
                    ],
                    "replan_count": replan_count,
                    "task_results": task_results
                }
                
        except Exception as e:
            print(f"Error in joiner: {str(e)}")
            error_message = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
            if task_content:  # task_results가 있다면 포함
                error_message += f"\n\n참고한 정보:\n{task_content}"
            return {
                "messages": [AIMessage(content=error_message)],
                "replan_count": replan_count + 1,
                "task_results": task_results
            }
    
    return joiner

# 테스트용 코드는 if __name__ == "__main__": 블록으로 이동
if __name__ == "__main__":
    async def main():
        messages = [
            {
                "role": "user",
                "content": "삼성전자의 2021년부터 2023년까지 영업이익 평균을 계산해줘"
            }
        ]
        
        try:
            result = await structured_llm_call(messages)
            print("Structured Output:", result.json(indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error in main: {str(e)}")

    asyncio.run(main()) 