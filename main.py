from dotenv import load_dotenv
load_dotenv()

from graph import app
from state import AgentState

if __name__ == "__main__":
    print("🚀 Starting Daily Research Agent...\n")

    # Task cố định cho scheduler
    task = "Summarize the latest developments in Agentic AI and LangGraph in 2026. Keep it short and clear."

    config = {"configurable": {"thread_id": "daily-run"}}

    try:
        final_answer = None
        
        for chunk in app.stream({"task": task}, config, stream_mode="values"):
            if chunk.get("messages"):
                last_msg = chunk.get("messages")[-1]
                if hasattr(last_msg, "content"):
                    print(f"→ {last_msg.content[:400]}...\n")

            if chunk.get("final_answer"):
                final_answer = chunk.get("final_answer")

        print("\n✅ FINAL ANSWER:")
        print(final_answer if final_answer else "No final answer generated.")
        print("\nWorkflow completed successfully!")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise e
