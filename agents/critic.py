from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",      # Stable & recommended model
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_retries=5,                 # Tự retry built-in của LangChain
    timeout=60                     # Timeout để tránh treo lâu
)

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    retry=retry_if_exception_type(ServerError)
)
def critic_node(state: AgentState):
    # Lấy lịch sử gần nhất để critic review
    messages = "\n".join([m[1] if isinstance(m, tuple) else str(m) for m in state.messages[-6:]])

    prompt = f"""You are the Critic agent.
Task: {state.task}

Recent outputs:
{messages}

Evaluate if the information is accurate, complete, and high-quality.
- If it is GOOD and ready for final answer → reply exactly "GOOD"
- If needs improvement → give clear, specific feedback on what to fix.

Only respond with feedback or "GOOD"."""

    try:
        result = llm.invoke([
            ("system", prompt),
            ("user", "Evaluate the latest output now.")
        ])

        content = result.content if hasattr(result, 'content') else str(result)

        if "GOOD" in content.upper():
            return {
                "next": "executor",
                "messages": [("assistant", "Critic: GOOD - Quality approved, proceed to final synthesis")]
            }
        else:
            return {
                "next": "supervisor",
                "messages": [("assistant", f"Critic: {content}")],
                "reflection_count": state.reflection_count + 1
            }
    except Exception as e:
        print(f"❌ Critic retry failed: {str(e)}")
        raise  # LangGraph sẽ catch và log
