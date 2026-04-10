from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import BaseModel
from state import AgentState

# Sử dụng Google Gemini - lấy key từ GitHub Secrets
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")   # Dòng này quan trọng
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
