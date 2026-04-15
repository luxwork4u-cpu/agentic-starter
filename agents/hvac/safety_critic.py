from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def safety_critic_node(state: AgentState):
    task = state.get("task") if isinstance(state, dict) else getattr(state, "task", "")
    messages = "\n".join([m[1] if isinstance(m, tuple) else str(m) for m in state.get("messages", [])[-5:]])

    prompt = f"""Bạn là Safety Critic chuyên về HVAC.
Triệu chứng: {task}

Recent analysis:
{messages}

Kiểm tra rủi ro an toàn:
- Có gas leak, electrical hazard, CO, high pressure không?
- Có cần cảnh báo "Tắt nguồn ngay" hoặc "Gọi pro ngay" không?

Trả lời ngắn gọn và kết thúc bằng "SAFE" hoặc "UNSAFE + lý do"."""

    result = llm.invoke([("system", prompt), ("user", "Đánh giá an toàn ngay")])

    content = result.content if hasattr(result, 'content') else str(result)

    if "SAFE" in content.upper():
        return {"next": "executor", "messages": [("assistant", "Safety Critic: SAFE - Có thể tiếp tục chẩn đoán")]}
    else:
        return {
            "next": "executor",
            "messages": [("assistant", f"Safety Critic: {content}")],
            "reflection_count": state.get("reflection_count", 0) + 1
        }