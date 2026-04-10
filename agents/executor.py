from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def executor_node(state: AgentState):
    result = llm.invoke([
        ("system", """You are the Executor. 
Your job is to synthesize all the information from previous agents into one clear, concise, and final answer.
Do not add new information. Just summarize and conclude based on what has been discussed."""),
        ("user", "\n".join([m.content for m in state.messages]))
    ])

    return {
        "messages": result,
        "final_answer": result.content,
        "next": "__end__"
    }
