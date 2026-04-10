from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=None
)

def executor_node(state: AgentState):
    result = llm.invoke([
        ("system", "You are the Executor. Synthesize all the information and give a clear, final answer."),
        ("user", "\n".join([m.content for m in state.messages]))
    ])

    return {
        "messages": result,
        "final_answer": result.content,
        "next": "__end__"
    }
