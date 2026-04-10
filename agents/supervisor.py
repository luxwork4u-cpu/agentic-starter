from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",      # Stable model hiện tại
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

class Route(BaseModel):
    next: str
    reasoning: str

def supervisor_node(state: AgentState):
    system = """You are the Supervisor. 
Route the task to the correct agent.
Available: researcher, critic, executor.
Use '__end__' only when the final answer is ready.
Return only valid JSON."""

    response = llm.with_structured_output(Route).invoke([
        ("system", system),
        ("user", f"Current task: {state.task}\nReflection count: {state.reflection_count}")
    ])

    return {
        "next": response.next,
        "messages": [("assistant", f"Supervisor: {response.reasoning}")]
    }
