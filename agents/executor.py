from langchain_openai import ChatOpenAI
from state import AgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
