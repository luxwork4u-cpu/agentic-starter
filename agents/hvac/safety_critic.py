from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def safety_critic_node(state: AgentState):
    messages = "\n".join([m[1] if isinstance(m, tuple) else str(m) for m in state.messages[-5:]])

    prompt = f"""Bạn là Safety Critic chuyên về HVAC.
Đánh giá chẩn đoán hiện tại cho triệu chứng: {state.task}

Recent analysis:
{messages}

Kiểm tra:
- Có rủi ro an toàn không? (gas leak, electrical hazard, CO, high pressure...)
- Có cần cảnh báo "Tắt nguồn ngay" hoặc "Gọi pro ngay" không?
- Chẩn đoán có an toàn để technician tự làm không?

Nếu an toàn và chẩn đoán hợp lý → trả lời "SAFE"
Nếu có nguy hiểm → đưa cảnh báo rõ ràng và gợi ý dừng lại.

Chỉ trả lời ngắn gọn + "SAFE" hoặc "UNSAFE"."""

    result = llm.invoke([("system", prompt), ("user", "Đánh giá an toàn ngay")])

    content = result.content if hasattr(result, 'content') else str(result)

    if "SAFE" in content.upper():
        return {"next": "executor", "messages": [("assistant", "Safety Critic: SAFE - Có thể tiếp tục chẩn đoán và sửa chữa")]}
    else:
        return {
            "next": "executor",
            "messages": [("assistant", f"Safety Critic: {content}")],
            "reflection_count": state.reflection_count + 1
        }