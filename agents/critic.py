from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_retries=6,
    timeout=90
)

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    retry=retry_if_exception_type(ServerError)
)
def critic_node(state: AgentState):
    messages = "\n".join([m[1] if isinstance(m, tuple) else str(m) for m in state.messages[-6:]])

    prompt = f"""You are the Critic agent.
Task: {state.task}

Recent outputs:
{messages}

Evaluate quality:
- If the information is useful, accurate, and the best we can get (especially if researcher says future info is impossible) → reply exactly "GOOD"
- Otherwise give specific feedback what to improve.

Do NOT approve empty or useless responses."""

    result = llm.invoke([
        ("system", prompt),
        ("user", "Evaluate now and be strict.")
    ])

    content = result.content if hasattr(result, 'content') else str(result)

    if "GOOD" in content.upper():
        return {
            "next": "executor",
            "messages": [("assistant", "Critic: GOOD - Ready for final synthesis")]
        }
    else:
        return {
            "next": "supervisor",
            "messages": [("assistant", f"Critic: {content}")],
            "reflection_count": state.reflection_count + 1
        }
