from langchain_openai import ChatOpenAI
from tools.web_search import web_search_tool
from state import AgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def researcher_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are the Researcher. Gather accurate, up-to-date information using tools."),
        ("user", state.task)
    ], tools=[web_search_tool])
    return {"messages": result, "next": "supervisor"}
