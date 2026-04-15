from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def executor_node(state: AgentState):
    task = state.get("task") if isinstance(state, dict) else getattr(state, "task", "")
    messages = "\n".join([m[1] if isinstance(m, tuple) else str(m) for m in state.get("messages", [])])

    prompt = f"""Bạn là Final Executor - HVAC Master Technician.
Tổng hợp chẩn đoán cho triệu chứng: {task}

Cung cấp báo cáo cuối cùng theo cấu trúc:
1. Tóm tắt vấn đề
2. Nguyên nhân chính xác nhất
3. Thứ tự kiểm tra & sửa chữa (bước 1, 2, 3...)
4. Tool cần chuẩn bị
5. Cảnh báo an toàn quan trọng
6. Khi nào nên gọi chuyên gia

Viết ngắn gọn, thực tế, dễ làm theo."""

    result = llm.invoke([("system", prompt), ("user", "Tạo báo cáo chẩn đoán cuối cùng")])

    final_answer = result.content if hasattr(result, 'content') else str(result)

    return {
        "final_answer": final_answer,
        "messages": [("assistant", f"Final Diagnostic Report:\n{final_answer}")],
        "next": "__end__"
    }