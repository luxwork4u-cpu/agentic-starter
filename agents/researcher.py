
from langchain_groq import ChatGroq
from tools.web_search import web_search_tool
from state import AgentState

# Dùng Groq
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

def researcher_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are the Researcher. Gather accurate, up-to-date information using tools."),
        ("user", state.task)
    ], tools=[web_search_tool])

    return {
        "messages": result,
        "next": "supervisor"
    }
