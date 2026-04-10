from langchain_google_genai import ChatGoogleGenerativeAI
from tools.web_search import web_search_tool
from state import AgentState

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key="AIzaSyCc5AQX2zpXMk-zhVErMbeL2KdhCjVmrcE"   # ← THAY BẰNG KEY THẬT CỦA BẠN
)

def researcher_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are the Researcher. Gather accurate, up-to-date information using tools if needed."),
        ("user", state.task)
    ], tools=[web_search_tool])

    return {
        "messages": result,
        "next": "supervisor"
    }
