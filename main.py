
import os
from datetime import datetime
from graph import app
from langgraph.checkpoint.memory import MemorySaver

print("🚀 Starting Daily Research Agent...")

# Dynamic task - tránh hỏi tương lai 2026 gây loop
current_date = datetime.now().strftime("%Y-%m-%d")
task = f"""
Summarize the latest developments in Agentic AI and LangGraph as of {current_date}.
Focus only on real, verifiable information available today.
Do not speculate about future years like 2026 or any unconfirmed developments.
Provide clear, structured summary with key points.
"""

# Config với recursion_limit để chống infinite loop
config = {
    "recursion_limit": 15,           # Tối đa 15 bước - an toàn cho GitHub Actions
    "configurable": {"thread_id": "daily-research-2026"}
}

# Memory saver (giữ state nếu cần)
checkpointer = MemorySaver()

try:
    final_answer = None
    print("🔄 Starting agent workflow...\n")

    for chunk in app.stream({"task": task}, config, stream_mode="values"):
        # In ra tiến trình để dễ debug
        if "messages" in chunk and chunk["messages"]:
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, tuple):
                print(f"→ {last_msg[0].capitalize()}: {last_msg[1][:200]}..." if len(str(last_msg[1])) > 200 else f"→ {last_msg[0].capitalize()}: {last_msg[1]}")
            else:
                print(f"→ Chunk: {str(last_msg)[:150]}...")

        # Lấy final_answer khi có
        if "final_answer" in chunk and chunk["final_answer"]:
            final_answer = chunk["final_answer"]
            print("\n✅ FINAL ANSWER READY!")
            break

    if final_answer:
        print("\n" + "="*80)
        print("📋 FINAL REPORT:")
        print("="*80)
        print(final_answer)
        print("="*80)
    else:
        print("\n⚠️  Workflow completed but no final_answer found.")

except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    print("💡 Tip: This is often temporary Google 503 overload. Try running again in a few minutes.")

print("\n🏁 Daily Research Agent finished.")
