from typing import Annotated, Literal
from pydantic import BaseModel
from langgraph.graph import add_messages

class AgentState(BaseModel):
    messages: Annotated[list, add_messages]
    next: Literal["researcher", "critic", "executor", "supervisor", "__end__"] = "supervisor"
    task: str
    reflection_count: int = 0
    max_reflections: int = 3
    final_answer: str | None = None
