
from langchain_groq import ChatGroq
from pydantic import BaseModel
from state import AgentState

# Sử dụng Groq với model mới
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

class Route(BaseModel):
    next: str
    reasoning: str

def supervisor_node(state: AgentState):
    system = """You are the Supervisor. Route the task to the right agent.
Available agents: researcher, critic, executor.
Finish with __end__ when you have a solid final answer.
Only output valid JSON with 'next' and 'reasoning'."""

    response = llm.with_structured_output(Route).invoke([
        ("system", system),
        ("user", f"Current task: {state.task}\nReflection count: {state.reflection_count}")
    ])

    return {
        "next": response.next,
        "messages": [("assistant", f"Supervisor routing to {response.next}: {response.reasoning}")]
    }
