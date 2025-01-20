import logging
import os
import json
import aiohttp
from typing import Dict, Any, Optional
from pydantic import BaseModel
from dataclasses import dataclass, fields
from plan.models import TaskPlan, PlanResult, Action, JoinResult
from dataclasses import dataclass
from typing import Literal, Union
import re
import traceback
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    RetryError
)

# 로거 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class HyperCLOVA:
    def __init__(self):
        self.url = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"
        self.headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": os.getenv("CLOVA_API_KEY"),
            "X-NCP-APIGW-API-KEY": os.getenv("CLOVA_API_GATEWAY_KEY"),
            "Content-Type": "application/json"
        }
        
    def _create_system_prompt(self, output_format: Dict[str, Any], additional_instructions: str = "") -> str:
        base_prompt = f"""
        
{additional_instructions}       
다음 JSON 형식으로만 응답해주세요:
{json.dumps(output_format, indent=2, ensure_ascii=False)}


주의사항:
1. 반드시 위 JSON 형식을 따라주세요
2. JSON 형식 외의 다른 텍스트는 포함하지 마세요
3. 모든 필드를 반드시 포함해주세요
4. 반드시 한국어로 답변해주세요
5. Action: 접두어 붙이지 마세요"""
        return base_prompt
    
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=5, min=5, max=400),
        retry=retry_if_exception_type((ValueError, Exception)),
        before_sleep=lambda retry_state: print(f"\n=== Rate limit 재시도 #{retry_state.attempt_number} ===\n"
                                     f"다음 시도까지 {retry_state.next_action.sleep} 초 대기...")
    )
    async def call_api(self, 
                      messages: list,
                      output_format: Dict[str, Any],
                      system_instructions: str = "",
                      format_class: Optional[type] = None,
                      temperature: float = 1) -> dataclass:
        """HyperCLOVA API 호출 함수"""
        
        system_prompt = self._create_system_prompt(output_format, system_instructions)
        messages = [{"role": "system", "content": system_prompt}, *messages]
        
        request_data = {
            "messages": messages,
            "topP": 0.8,
            "topK": 5,
            "maxTokens": 4000,
            "temperature": temperature,
            "repeatPenalty": 5.0,
            "stopBefore": [],
            "includeAiFilters": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=request_data, headers=self.headers) as response:
                    result = await response.json()
                    
                    # Rate limit 체크를 여기서 먼저 수행
                    if "error" in result and "Rate limit exceeded" in str(result.get("error")):
                        raise ValueError("Rate limit exceeded")
                        
                    if format_class:
                        return self._parse_response(result, format_class)
                    else:
                        return result
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                raise ValueError(f"Rate limit exceeded: {str(e)}")
            raise Exception(f"API 호출 중 오류 발생: {str(e)}")

    def _parse_response(self, result: Dict[str, Any], format_class: type) -> Any:
        """API 응답을 주어진 dataclass 형식에 맞게 파싱하는 함수"""
        try:
            print(f"\n=== _parse_response input logging : {result}")
            
            # rate limit 에러 체크
            if result.get('status', {}).get('code') == '42901':
                raise ValueError(f"Rate limit exceeded: {result['status'].get('message')}")

            content_str = result.get('result', {}).get('message', {}).get('content', '{}')
            print(f"=== Content String 디버깅 ===\n{content_str}")
            if format_class.__name__ == 'PlanResult':
                try:
                    # 먼저 JSON 파싱 시도
                    content_json = json.loads(content_str)
                    task_plans = []
                    for task in content_json.get('tasks', []):
                        if task.get('tool') and task.get('tool') != 'join':
                            task_plans.append(TaskPlan(
                                idx=task['idx'],
                                tool=task['tool'],
                                args=task['args'],
                                dependencies=task.get('dependencies', [])
                            ))
                    
                    return PlanResult(
                        thought=content_json.get('thought', "작업 계획 수립 완료"),
                        tasks=task_plans
                    )
                    
                except json.JSONDecodeError:
                    # JSON 파싱 실패시 텍스트 기반 파싱
                    tasks = []
                    lines = content_str.split('\n')
                    current_task = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # 새로운 태스크 시작 (숫자. 으로 시작)
                        task_match = re.match(r'(\d+)\.\s+(.+)', line)
                        if task_match:
                            if current_task:
                                tasks.append(current_task)
                            idx = int(task_match.group(1)) - 1
                            current_task = {
                                'idx': idx,
                                'tool': None,
                                'args': {},
                                'dependencies': [idx-1] if idx > 0 else []
                            }
                            
                        # tool 정보
                        elif line.strip().startswith('- tool :'):
                            tool = line.split(':')[1].strip()
                            if current_task:
                                current_task['tool'] = tool
                                
                        # args 정보
                        elif line.strip().startswith('- args :'):
                            args_str = line.split(':')[1].strip()
                            args = {}
                            for arg in args_str.split(','):
                                if '=' in arg:
                                    key, value = arg.split('=', 1)
                                    args[key.strip()] = value.strip()
                            if current_task:
                                current_task['args'] = args
                                
                    # 마지막 태스크 추가
                    if current_task:
                        tasks.append(current_task)
                        
                    task_plans = []
                    for task in tasks:
                        if task['tool'] and task['tool'] != 'join':
                            task_plans.append(TaskPlan(
                                idx=task['idx'],
                                tool=task['tool'],
                                args=task['args'],
                                dependencies=task['dependencies']
                            ))
                    
                    print(f"Parsed tasks: {task_plans}")
                    return PlanResult(
                        thought="작업 계획 수립 완료",
                        tasks=task_plans
                    )

            elif format_class.__name__ == 'JoinResult':

                content_obj = None
                
                try:
                    content_str = content_str.strip()
                    print(f"Processing content string: {content_str}")
                    
                    # Case 1: "Action: {json}" 형식
                    if content_str.startswith('Action:'):
                        content_str = content_str[7:].strip()
                        
                        # JSON 형식인 경우
                        if content_str.startswith('{'):
                            try:
                                content_obj = json.loads(content_str)
                            except json.JSONDecodeError:
                                pass
                    
                    # Case 2: "Action: replan\nContent: {json}" 형식
                    elif '\n' in content_str and content_str.startswith('Action:'):
                        action_line, content_line = content_str.split('\n', 1)
                        action_type = action_line[7:].strip()
                        
                        if content_line.startswith('Content:'):
                            try:
                                content_json = content_line[8:].strip()
                                content_data = json.loads(content_json)
                                content_obj = {
                                    'type': action_type,
                                    'content': {
                                        action_type: {
                                            'feedback' if action_type == 'replan' else 'response': 
                                            content_data.get('message', '')
                                        }
                                    }
                                }
                            except json.JSONDecodeError:
                                pass
                    
                    # Case 3: 일반 JSON 형식
                    if content_obj is None and content_str.startswith('{'):
                        try:
                            content_obj = json.loads(content_str)
                        except json.JSONDecodeError:
                            pass
                    
                    print(f"Parsed content object: {content_obj}")
                    
                    # 파싱된 객체 처리
                    if content_obj:
                        # 최상위 레벨에 type이 있는 경우
                        if 'type' in content_obj:
                            content_obj = {
                                'thought': '',
                                'action': content_obj
                            }
                        
                        if 'action' in content_obj:
                            action_obj = content_obj['action']
                            action_type = action_obj.get('type', '')
                            
                            if action_type == 'final_response':
                                response = (
                                    action_obj.get('content', {})
                                    .get('final_response', {})
                                    .get('response', '') or 
                                    action_obj.get('content', {}).get('message', '') or 
                                    "죄송합니다. 현재 답변이 어렵습니다."
                                )
                                return JoinResult(
                                    thought=content_obj.get('thought', ''),
                                    action=Action(
                                        type='final_response',
                                        response=response
                                    )
                                )
                            elif action_type == 'replan':
                                feedback = (
                                    action_obj.get('content', {}).get('replan', {}).get('feedback') or
                                    action_obj.get('content', {}).get('message', '') or
                                    "재계획이 필요합니다."
                                )
                                return JoinResult(
                                    thought=content_obj.get('thought', ''),
                                    action=Action(
                                        type='replan',
                                        feedback=feedback
                                    )
                                )
                            else:
                                raise ValueError(f"Unknown action type: {action_type}")
                    
                    # 파싱 실패한 경우 원본 텍스트 반환
                    return JoinResult(
                        thought="",
                        action=Action(
                            type="final_response",
                            response=content_str
                        )
                    )    
                except Exception as e:
                    print(f"Join parsing error: {str(e)}")
                    return JoinResult(
                        thought="",
                        action=Action(
                            type="final_response",
                            response=content_str
                        )
                    )            # PlanResult가 아닌 경우의 처리
            try:
                return format_class(**json.loads(content_str))
            except json.JSONDecodeError:
                return format_class(**{"content": content_str})
                
        except Exception as e:
            print(f"\n=== [HYPERCLOVA] Response parsing error: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise ValueError(f"응답 파싱 중 오류 발생: {str(e)}")
    def _extract_field_value(self, content_obj: Dict[str, Any], field_name: str, field_type: type) -> Any:
        """content_obj에서 특정 필드의 값을 추출하는 헬퍼 함수"""
        try:
            # 중첩된 객체 처리
            if hasattr(field_type, '__dataclass_fields__'):
                nested_data = {}
                nested_fields = {field.name: field.type for field in fields(field_type)}
                
                # 중첩된 객체의 각 필드에 대해 재귀적으로 처리
                for nested_name, nested_type in nested_fields.items():
                    nested_value = self._extract_field_value(
                        content_obj[field_name]['content'][field_name], 
                        nested_name, 
                        nested_type
                    )
                    nested_data[nested_name] = nested_value
                
                return field_type(**nested_data)
            
            # 기본 필드 처리
            if field_name in content_obj:
                return content_obj[field_name]
            # content 내부 검색
            elif 'content' in content_obj and field_name in content_obj['content']:
                return content_obj['content'][field_name]
            else:
                raise KeyError(f"Field '{field_name}' not found in response")
            
        except (KeyError, TypeError) as e:
            raise ValueError(f"필드 '{field_name}' 추출 중 오류 발생: {e}") 