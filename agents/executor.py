from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def executor_node(state: AgentState):
    messages = "\n".join([m[1] if isinstance(m, tuple) else str(m) for m in state.messages])

    prompt = f"""You are the Executor / Final Synthesizer.
Task: {state.task}

All previous research and critique:
{messages}

Produce a clean, professional, well-structured final report.
Include key findings, sources if available, and clear conclusion."""

    result = llm.invoke([("system", prompt), ("user", "Generate final answer now")])

    final_answer = result.content if hasattr(result, 'content') else str(result)

    return {
        "final_answer": final_answer,
        "messages": [("assistant", f"Final Answer: {final_answer}")],
        "next": "__end__"
    }
