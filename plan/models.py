from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional, Union
from langchain.tools import BaseTool

@dataclass
class TaskPlan:
    idx: int
    tool: Union[str, BaseTool]
    args: Dict[str, Any]
    dependencies: List[int] = field(default_factory=list)
    thought: Optional[str] = None

@dataclass
class PlanResult:
    thought: str
    tasks: List[TaskPlan] 

@dataclass
class FinalResponseContent:
    response: str

@dataclass
class ReplanContent:
    feedback: str

@dataclass
class ActionContent:
    final_response: Optional[FinalResponseContent] = None
    replan: Optional[ReplanContent] = None

@dataclass
class Action:
    type: str
    response: Optional[str] = None
    feedback: Optional[str] = None

@dataclass
class JoinResult:
    thought: str
    action: Action