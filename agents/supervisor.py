from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from state import AgentState
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # Stable & recommended
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_retries=7,
    timeout=90
)

class Route(BaseModel):
    next: str
    reasoning: str

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=30),
    retry=retry_if_exception_type(ServerError)
)
def supervisor_node(state: AgentState):
    system = """You are the Supervisor.
If researcher says the information is unavailable or about the future (like 2026), DO NOT loop.
Route to executor for final synthesis with available info, or to __end__.

Otherwise:
- researcher → more info needed
- critic → check quality
- executor → final answer
- __end__ when done"""

    try:
        response = llm.with_structured_output(Route).invoke([
            ("system", system),
            ("user", f"Task: {state.task}\nRecent: {str(state.messages[-3:]) if state.messages else 'Start'}")
        ])
        return {
            "next": response.next,
            "messages": [("assistant", f"Supervisor: {response.reasoning}")]
        }
    except Exception as e:
        print(f"❌ Supervisor fallback: {type(e).__name__}")
        return {"next": "executor", "messages": [("assistant", "Supervisor: Using available information for final answer")]}
