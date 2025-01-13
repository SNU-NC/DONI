from typing import TypeVar, List, Dict, Any, TypedDict
from datetime import datetime

T = TypeVar("T")

class TaskResult(TypedDict):
    """Task 실행 결과를 저장하는 타입"""
    tool_name: str
    task_id: str
    timestamp: str
    result: Any
    metadata: Dict[str, Any]

def add_task_results(existing: List[TaskResult], new: List[TaskResult]) -> List[TaskResult]:
    """Task 결과를 누적하는 accumulator 함수"""
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new

def create_task_result(
    tool_name: str,
    task_id: str,
    result: Any,
    metadata: Dict[str, Any] = None
) -> TaskResult:
    """TaskResult 객체를 생성하는 헬퍼 함수"""
    return {
        "tool_name": tool_name,
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "metadata": metadata or {}
    }

def format_reference_for_response(task_results: List[TaskResult]) -> str:
    """Task 결과를 응답 형식에 맞게 포맷팅"""
    if not task_results:
        return "참고 데이터 없음"
        
    formatted_refs = []
    for result in task_results:
        ref = f"- {result['tool_name']} ({result['timestamp']})"
        if result['metadata']:
            meta_str = ", ".join(f"{k}: {v}" for k, v in result['metadata'].items())
            ref += f"\n  메타데이터: {meta_str}"
        formatted_refs.append(ref)
        
    return "\n".join(formatted_refs) 