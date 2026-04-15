from graph_hvac import hvac_app
from datetime import datetime

print("🔧 HVAC Troubleshooting Agent - Multi-Agent System")
print("=" * 75)

print("Nhập triệu chứng HVAC của bạn (bằng tiếng Việt hoặc English):")
symptoms = input("\n→ Triệu chứng: ")

if not symptoms.strip():
    print("Vui lòng nhập triệu chứng!")
    exit()

task = f"Chẩn đoán và đưa ra hướng dẫn sửa chữa cho triệu chứng sau: {symptoms}"

# Khởi tạo state đúng format
initial_state = {
    "task": task,
    "messages": [],
    "next": "supervisor",      # Bắt đầu từ supervisor
    "reflection_count": 0,
    "final_answer": ""
}

config = {
    "recursion_limit": 12,
    "configurable": {"thread_id": f"hvac-{datetime.now().strftime('%Y%m%d-%H%M')}"}
}

print("\n🔍 Agent đang phân tích triệu chứng...\n")

try:
    final_answer = None
    for chunk in hvac_app.stream(initial_state, config, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, tuple):
                role = last_msg[0].replace("_", " ").title()
                content = str(last_msg[1])[:280] + "..." if len(str(last_msg[1])) > 280 else str(last_msg[1])
                print(f"→ {role}: {content}")

        if "final_answer" in chunk and chunk["final_answer"]:
            final_answer = chunk["final_answer"]
            break

    if final_answer:
        print("\n" + "="*80)
        print("📋 BÁO CÁO CHẨN ĐOÁN HVAC CUỐI CÙNG")
        print("="*80)
        print(final_answer)
        print("="*80)
    else:
        print("⚠️ Không tìm thấy final answer.")

except Exception as e:
    print(f"❌ Lỗi: {str(e)}")

print("\n🏁 HVAC Troubleshooting Agent finished.")