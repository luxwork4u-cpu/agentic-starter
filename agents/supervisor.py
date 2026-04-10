
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from state import AgentState

# Sử dụng Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   # Model miễn phí, nhanh và khá mạnh
    temperature=0,
    google_api_key=None   # Sẽ tự lấy từ GEMINI_API_KEY
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
