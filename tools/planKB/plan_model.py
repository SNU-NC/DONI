from pydantic import BaseModel
from typing import List , Optional
class Plan(BaseModel):

    steps: List[str]
class PlanExample(BaseModel):
    query: str
    plan: Plan 
    extra_info: Optional[str] = None