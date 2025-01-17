import os
import sys

# 현재 파일의 절대 경로
current_path = os.getcwd()

# 프로젝트의 루트 디렉토리 (현재 경로의 상위 디렉토리)
project_root = os.path.abspath(os.path.join(current_path, ".."))
# 마지막 디렉터리 이름을 추출
last_directory_name = os.path.basename(project_root)

# PYTHONPATH에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.api_config import (
    CLOVA_API_KEY,
    CLOVA_API_GATEWAY_KEY,
    OPENAI_API_KEY
)
from config.prompts import _math_tool_DESCRIPTION
# API 키와 Gateway API 키를 넣습니다.

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from tools.math.math_tools import get_math_tool
from langchain_community.chat_models import ChatClovaX
from tools.financialTerm.fin_knowledge_tools import get_fin_tool
from tools.retrieve.analysistReport.report_RAG_Tool import RAGTool
from langchain_core.tools import BaseTool

load_dotenv()   
NCP_CLOVASTUDIO_API_KEY = os.getenv("NCP_CLOVASTUDIO_API_KEY")
NCP_APIGW_API_KEY = os.getenv("NCP_APIGW_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

calculate = get_math_tool(ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY))
retriever_tool = RAGTool()
tavily_tool = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
    api_key=TAVILY_API_KEY
)

fin_knowledge = get_fin_tool(ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY))

from langgraph.graph import MessagesState

class AgentState(MessagesState):
    # The 'next' field indicates where to route to next
    next : str

from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import Optional

members = ["researcher", "calculator", "retriever"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. First Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH. Prioritize the retriever over the researcher."
    " If the retriever doesn't provide any information, then use the researcher."
    " When you ask to retirever, make query simple to understand for retrieving"
)

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

def supervisor_node(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END
    return {"next": next_}

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(
    llm, tools=[tavily_tool], state_modifier="You are a researcher. DO NOT do any math. Just give required data to supervisor"
)

def research_node(state: AgentState) -> AgentState:
    result = research_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="researcher")
        ]
    }

retrieve_agent = create_react_agent(
    llm, tools=[retriever_tool], state_modifier="You can find data from documnets. DO NOT do any math. Give required data to supervisor"
)

def retrieve_node(state: AgentState) -> AgentState:
    result = retrieve_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="retriever")
        ]
    }

calculate_agent = create_react_agent(
    llm,
    tools = [calculate],
    state_modifier= make_system_prompt(
        "You can only calcuate. Give calculated answer to supervisor"
    )
)

def calculate_node(state: AgentState) -> AgentState:
    result = calculate_agent.invoke(state)
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="calculator"
    )
    return {
        "messages": result["messages"],
    }

knowledge_agent = create_react_agent(
    llm,
    tools= [fin_knowledge],
    state_modifier="You are knowledge finder. Give formula to superviosr"
)

def knowledge_node(state: AgentState) -> AgentState:
    result = knowledge_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="fin_knowledge")
        ]
    }

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("calculator", calculate_node)
builder.add_node("retriever", retrieve_node)

for member in members:
    builder.add_edge(member, "supervisor")

builder.add_conditional_edges("supervisor", lambda state: state["next"])

builder.add_node("fin_knowledge", knowledge_node)
builder.add_edge(START, "fin_knowledge")
builder.add_edge("fin_knowledge", "supervisor")

graph = builder.compile()

class CalculatorGraphInput(BaseModel):
    """계산기 그래프 입력 스키마"""
    query: str
    context: Optional[list] = None

class CalculatorGraphTool(BaseTool):
    name: str = "calculator_graph_tool"
    description: str = _math_tool_DESCRIPTION
    args_schema: type[BaseModel] = CalculatorGraphInput
    graph: StateGraph = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.graph = self._initialize_graph()

    def _initialize_graph(self) -> StateGraph:
        """StateGraph 초기화"""
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("researcher", research_node)
        builder.add_node("calculator", calculate_node)
        builder.add_node("retriever", retrieve_node)
        
        builder.add_edge(START, "supervisor")

        for member in ["researcher", "calculator", "retriever"]:
            builder.add_edge(member, "supervisor")
            
        builder.add_conditional_edges(
            "supervisor", 
            lambda state: state["next"]
        )
        
        return builder.compile()

    async def _arun(self, query: str, context: Optional[list] = None) -> str:
        """비동기 실행"""
        messages = [("user", query)]
        if context:
            messages.extend([("system", ctx) for ctx in context])

        final_result = None
        async for step in self.graph.astream(
            {"messages": messages}, 
            subgraphs=False
        ):
            final_result = step
            
        if final_result and "messages" in final_result:
            return final_result["messages"][-1].content
        return "계산을 완료할 수 없습니다."

    def _run(self, query: str, context: Optional[list] = None) -> str:
        """동기 실행"""
        messages = [("user", query)]
        if context:
            messages.extend([("system", ctx) for ctx in context])

        final_result = None
        for step in self.graph.stream(
            {"messages": messages}, 
            subgraphs=False
        ):
            print(step)
            final_result = step
            
        if final_result and "messages" in final_result:
            return final_result["messages"][-1].content
        return "계산을 완료할 수 없습니다."