from langchain_google_genai import ChatGoogleGenerativeAI
from tools.web_search import web_search_tool
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="hướng dẫn bạn sửa nhanh các file còn lại (researcher, critic, executor)",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def researcher_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are the Researcher. Gather accurate, up-to-date information about the given task. Use the web search tool if needed."),
        ("user", state.task)
    ], tools=[web_search_tool])

    return {
        "messages": result,
        "next": "supervisor"
    }
