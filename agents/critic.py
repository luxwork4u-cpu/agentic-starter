
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def critic_node(state: AgentState):
    messages = "\n".join([m[1] if isinstance(m, tuple) else str(m) for m in state.messages[-5:]])

    prompt = f"""You are the Critic agent.
Review the latest research output for the task: {state.task}

Current content:
{messages}

Is the information accurate, complete, and well-structured?
If YES and ready for final answer → respond with "GOOD"
If needs improvement → give specific feedback what to fix.

Only respond with clear feedback or "GOOD"."""

    result = llm.invoke([("system", prompt), ("user", "Evaluate now")])

    content = result.content if hasattr(result, 'content') else str(result)

    if "GOOD" in content.upper():
        return {"next": "executor", "messages": [("assistant", "Critic: GOOD - Ready for final synthesis")]}
    else:
        return {
            "next": "supervisor",
            "messages": [("assistant", f"Critic: {content}")],
            "reflection_count": state.reflection_count + 1
        }
