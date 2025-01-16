import os
import json
import aiohttp
from typing import Dict, Any, Optional
from pydantic import BaseModel
from dataclasses import dataclass, fields



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
4. 반드시 한국어로 답변해주세요"""
        return base_prompt

    async def call_api(self, 
                      messages: list,
                      output_format: Dict[str, Any],
                      system_instructions: str = "",
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
                    return self._parse_response(result)
        except Exception as e:
            raise Exception(f"API 호출 중 오류 발생: {str(e)}")

    def _parse_response(self, result: Dict[str, Any], format_class: type) -> Any:
        """API 응답을 주어진 dataclass 형식에 맞게 파싱하는 함수"""
        if not (result and 'result' in result and 'message' in result['result']):
            raise ValueError("Invalid API response format")
        
        try:
            # API 응답에서 content 추출 및 파싱
            content_str = result['result']['message']['content']
            content_obj = json.loads(content_str)
            
            # dataclass의 필드 정보 가져오기
            from dataclasses import fields
            format_fields = {field.name: field.type for field in fields(format_class)}
            
            # 결과 데이터를 저장할 딕셔너리
            parsed_data = {}
            
            # 각 필드에 대해 데이터 추출
            for field_name, field_type in format_fields.items():
                value = self._extract_field_value(content_obj, field_name, field_type)
                parsed_data[field_name] = value
            
            # dataclass 인스턴스 생성 및 반환
            return format_class(**parsed_data)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"응답 파싱 중 오류 발생: {e}")
    
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