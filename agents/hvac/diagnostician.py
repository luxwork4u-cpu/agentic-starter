from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def diagnostician_node(state: AgentState):
    prompt = f"""Bạn là Diagnostician HVAC có 15 năm kinh nghiệm.
Triệu chứng từ khách hàng: {state.task}

Phân tích chi tiết:
- Các nguyên nhân phổ biến nhất theo thứ tự khả năng cao → thấp
- Yếu tố có thể gây ra triệu chứng (điện, gas, refrigerant, sensor, motor, board...)
- Dấu hiệu phân biệt giữa các nguyên nhân

Trả lời ngắn gọn, kỹ thuật nhưng dễ hiểu."""

    result = llm.invoke([("system", prompt), ("user", "Phân tích nguyên nhân ngay")])

    content = result.content if hasattr(result, 'content') else str(result)

    return {
        "messages": [("assistant", f"Diagnostician: {content}")],
        "next": "supervisor"
    }