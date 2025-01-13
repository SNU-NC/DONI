import logging
from typing import List, Union, Dict, Any
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import TypedDict

class FinalResponse(BaseModel):
    """The final response/answer."""
    response: str = Field(description="사용자 질문에 대한 상세한 답변")


class JoinerState(TypedDict):
    messages: List[BaseMessage]
    replan_count: int
    force_final_answer: bool

class Replan(BaseModel):
    """Feedback for replanning."""
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decision model for replanning or final response."""
    thought: str = Field(
        description="The chain of thought reasoning for the selected action"

    )
    action: Union[FinalResponse, Replan]


def _parse_joiner_output(decision: JoinOutputs) -> Dict[str, Any]:
    """Parse the joiner output and convert to messages."""
    response = [AIMessage(content=f"Thought: {decision.thought}")]

    if isinstance(decision.action, Replan): 
        return {
            "messages": response + [SystemMessage(content=f"Context from last attempt: {decision.action.feedback}")],
        }
    else:

        formatted_response = decision.action.response
        
        # key_information이 있다면 State에 추가
        key_info = decision.action.key_information if hasattr(decision.action, 'key_information') else []
        
        return {
            "messages": response + [AIMessage(content=formatted_response)],
            "key_information": key_info  # key_information 포함
        }


def select_recent_messages(state) -> dict:
    """Select the most recent messages up to the last human message."""
    messages = state["messages"]
    replan_count = state["replan_count"]
    refference = state["task_results"]
    print("refference 확인 *************")
    for ref in refference:
        print(ref)
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    
    return {"messages": selected[::-1], "replan_count": replan_count }


def check_replan_count(state: dict):
    """Check replan count and decide next step."""
    messages = state["messages"]
    replan_count = state["replan_count"]
    
    if replan_count >= 2:
        logging.info(f"Replan count exceeded ({replan_count}), forcing final answer")
        return {
            "messages": messages,
            "replan_count": replan_count,
            "force_final_answer": True
        }
    logging.info(f"Current replan count: {replan_count}, continuing normal path")

    return {
        "messages": messages,
        "replan_count": replan_count,
        "force_final_answer": False
    }



def create_joiner(llm: BaseChatModel):
    """Create a joiner chain with the given LLM."""
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d")
    
    joiner_prompt = hub.pull("snunc/rag-llm-compiler-joiner-v2").partial(
        examples="", 
        time=current_time,
        format_instructions=""
    )

    final_prompt = hub.pull("snunc/rag-llm-compiler-final-answer").partial(
        examples="",
        time=current_time,
   )
    
    # 함수들을 Runnable로 변환
    select_messages_runnable = RunnableLambda(select_recent_messages)
    check_replan_runnable = RunnableLambda(check_replan_count)
    parse_output_runnable = RunnableLambda(_parse_joiner_output)
    
    # 경로 정의
    normal_path = joiner_prompt | llm.with_structured_output(JoinOutputs)
    replan_fail_path = final_prompt | llm.with_structured_output(JoinOutputs)
    
    def should_use_final_answer(state: Dict[str, Any]) -> bool:
        result = state["force_final_answer"]
        logging.info(f"Should use final answer? {result}")
        return state["force_final_answer"]
    
    final_answer_path = RunnableBranch(
        (should_use_final_answer, replan_fail_path),
        normal_path
    )
    
    # 체인 구성
    chain = (
        select_messages_runnable
        | check_replan_runnable
        | final_answer_path
        | parse_output_runnable
    )
    
    # 체인을 쓸지 말지 결정
    def conditional_joiner(state: Dict[str, Any]) -> Dict[str, Any]:
        report_agent_use = state.get("report_agent_use", False)  # 디폴트: False
        print("stop_joiner 값은 뭐야 ? ", report_agent_use)
        if report_agent_use:
            # 체인을 안 썼을 때 반환할 수 있는 간단한 결과 예시
            print("체인을 사용하지 않았습니다.")
            return {
                "messages": [AIMessage(content=state["messages"][-1].content)],
            }
        print("chain 씀")
        # use_joiner가 False라면 실제 체인 실행
        return chain.invoke(state)

    
    return conditional_joiner