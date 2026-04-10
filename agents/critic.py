from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def critic_node(state: AgentState):
    result = llm.invoke([
        ("system", """You are the Critic. Carefully review the current work.
Point out any gaps, hallucinations, contradictions, or missing information.
If the work is solid and complete, reply with the word 'GOOD' only and suggest moving to the executor."""),
        ("user", "\n".join([m.content for m in state.messages[-4:]]))
    ])

    state.reflection_count += 1
    return {
        "messages": result,
        "next": "supervisor" if "GOOD" not in result.content.upper() else "executor",
        "reflection_count": state.reflection_count
    }
