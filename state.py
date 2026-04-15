from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage
from pydantic import Field

# Cập nhật AgentState để hỗ trợ cả dự án cũ và HVAC agent mới
class AgentState(TypedDict):
    task: str
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    next: Literal[
        "researcher", "critic", "executor", "supervisor", "__end__",
        "diagnostician", "safety_critic"   # Thêm tên mới cho HVAC agent
    ]
    reflection_count: Annotated[int, "add"] = Field(default=0)
    final_answer: str = ""