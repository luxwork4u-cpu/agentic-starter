from langchain_groq import ChatGroq
from pydantic import BaseModel
from state import AgentState

# Sử dụng Groq - tự lấy key từ GitHub Secrets
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
    # Không cần ghi groq_api_key=None nữa, langchain_groq sẽ tự lấy từ GROQ_API_KEY
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
