from langchain_groq import ChatGroq
from state import AgentState

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

def critic_node(state: AgentState):
    result = llm.invoke([
        ("system", """You are the Critic. Review the current work and point out gaps, hallucinations, or missing information.
If everything looks solid, reply with 'GOOD' and suggest moving to executor."""),
        ("user", "\n".join([m.content for m in state.messages[-4:]]))
    ])

    state.reflection_count += 1
    return {
        "messages": result,
        "next": "supervisor" if "GOOD" not in result.content.upper() else "executor",
        "reflection_count": state.reflection_count
    }
