import os
from datetime import datetime
from graph import app

print("🚀 Starting Daily Research Agent...")

current_date = datetime.now().strftime("%Y-%m-%d")
task = f"""
Summarize the latest developments in Agentic AI and LangGraph as of {current_date}.
Focus only on real, verifiable information available today.
Do not speculate about any future dates.
"""

config = {
    "recursion_limit": 15,
    "configurable": {"thread_id": "daily-research"}
}

try:
    final_answer = None
    print("🔄 Starting agent workflow...\n")

    for chunk in app.stream({"task": task}, config, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            last = chunk["messages"][-1]
            if isinstance(last, tuple):
                print(f"→ {last[0].capitalize()}: {str(last[1])[:180]}..." if len(str(last[1])) > 180 else f"→ {last[0].capitalize()}: {last[1]}")

        if "final_answer" in chunk and chunk["final_answer"]:
            final_answer = chunk["final_answer"]
            print("\n✅ FINAL ANSWER READY!")
            break

    if final_answer:
        print("\n" + "="*80)
        print("📋 FINAL REPORT:")
        print("="*80)
        print(final_answer)

        # === LƯU OUTPUT RA FILE ===
        output_file = "daily_research_output.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Daily Research Agent Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(final_answer)
        
        print(f"\n💾 Output saved to → {output_file}")
    else:
        print("\n⚠️ No final_answer found.")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")

print("\n🏁 Daily Research Agent finished.")
