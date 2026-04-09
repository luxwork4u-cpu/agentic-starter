from dotenv import load_dotenv
load_dotenv()

from graph import app
from state import AgentState

if __name__ == "__main__":
    # Phiên bản đơn giản cho GitHub Actions (không cần input)
    task = "Compare the top 3 agentic frameworks in 2026 and recommend the best starter setup for a solo developer building production RAG + automation."

    print("\n🚀 Starting agentic workflow...\n")

    config = {"configurable": {"thread_id": "github-action-run"}}

    final_answer = None

    for chunk in app.stream({"task": task}, config, stream_mode="values"):
        last_msg = chunk.get("messages", [])[-1] if chunk.get("messages") else None
        if last_msg and hasattr(last_msg, "content"):
            print(f"[{chunk.get('next', 'unknown')}] {last_msg.content[:300]}...")

        if chunk.get("final_answer"):
            final_answer = chunk.get("final_answer")

    print("\n✅ FINAL ANSWER:")
    print(final_answer if final_answer else "No final answer generated.")
    
    print("\nWorkflow completed successfully!")
