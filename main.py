from dotenv import load_dotenv
load_dotenv()

from graph import app
from state import AgentState

if __name__ == "__main__":
    task = input("Enter your task (or press Enter for demo): ").strip()
    if not task:
        task = "Compare the top 3 agentic frameworks in 2026 and recommend the best starter setup for a solo developer building production RAG + automation."

    config = {"configurable": {"thread_id": "demo-001"}}

    print("\n🚀 Starting agentic workflow...\n")

    initial_state = AgentState(task=task)

    for chunk in app.stream(initial_state, config, stream_mode="values"):
        last_msg = chunk.get("messages", [])[-1] if chunk.get("messages") else None
        if last_msg and hasattr(last_msg, "content"):
            print(f"[{chunk.get('next', 'unknown')}] {last_msg.content[:200]}...")

    print("\n✅ FINAL ANSWER:")
    print(chunk.get("final_answer") or "No final answer generated.")
