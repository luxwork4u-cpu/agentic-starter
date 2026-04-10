from langchain_groq import ChatGroq
from state import AgentState

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def researcher_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are the Researcher. Gather accurate, up-to-date information about the task."),
        ("user", state.task)
    ])

    return {
        "messages": result,
        "next": "supervisor"
    }
