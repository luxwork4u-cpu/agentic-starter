from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def critic_node(state: AgentState):
    result = llm.invoke([
        ("system", """You are the Critic. Review the current work carefully.
Point out gaps, hallucinations, or missing information.
If everything looks solid and complete, reply with 'GOOD' and suggest moving to executor."""),
        ("user", "\n".join([m.content for m in state.messages[-4:]]))
    ])

    state.reflection_count += 1
    return {
        "messages": result,
        "next": "supervisor" if "GOOD" not in result.content.upper() else "executor",
        "reflection_count": state.reflection_count
    }
