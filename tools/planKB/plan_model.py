from pydantic import BaseModel
from typing import List
class Plan(BaseModel):
    level: int
    steps: List[str]
class PlanExample(BaseModel):
    query: str
    plan: Plan 
