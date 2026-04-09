from langchain_groq import ChatGroq
from state import AgentState

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

def executor_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are the Executor. Synthesize everything into a clear, final answer."),
        ("user", "\n".join([m.content for m in state.messages]))
    ])

    return {
        "messages": result,
        "final_answer": result.content,
        "next": "__end__"
    }
