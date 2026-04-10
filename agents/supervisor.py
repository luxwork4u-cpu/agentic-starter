from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

class Route(BaseModel):
    next: str
    reasoning: str

def supervisor_node(state: AgentState):
    system = """You are the Supervisor. Decide which agent should handle the task next.
Available agents: researcher, critic, executor.
Use '__end__' when the task is complete and ready for final answer.
Return only valid JSON with 'next' and 'reasoning'."""

    response = llm.with_structured_output(Route).invoke([
        ("system", system),
        ("user", f"Current task: {state.task}\nReflection count: {state.reflection_count}")
    ])

    return {
        "next": response.next,
        "messages": [("assistant", f"Supervisor: {response.reasoning}")]
    }
