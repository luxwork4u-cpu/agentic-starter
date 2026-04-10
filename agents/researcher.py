from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def researcher_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are a Researcher. Provide accurate, up-to-date information about the task."),
        ("user", state.task)
    ])

    return {
        "messages": result,
        "next": "supervisor"
    }
