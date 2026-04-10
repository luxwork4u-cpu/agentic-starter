from langchain_google_genai import ChatGoogleGenerativeAI
from tools.web_search import web_search_tool
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def researcher_node(state: AgentState):
    prompt = f"""You are the Researcher agent.
Task: {state.task}

Gather accurate, up-to-date information. Use the web_search tool if needed.
Provide detailed findings with sources if possible."""

    result = llm.invoke([
        ("system", prompt),
        ("user", state.task)
    ], tools=[web_search_tool])

    return {
        "messages": [("assistant", f"Researcher: {result.content if hasattr(result, 'content') else str(result)}")],
        "next": "supervisor"
    }
