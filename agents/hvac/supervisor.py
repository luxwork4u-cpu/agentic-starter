from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

class Route(BaseModel):
    next: str
    reasoning: str

def supervisor_node(state: AgentState):
    system = """Bạn là Supervisor của HVAC Troubleshooting Team.
Nhiệm vụ: Phân tích triệu chứng HVAC và route đến agent phù hợp.

Available agents:
- diagnostician: Thu thập và phân tích nguyên nhân kỹ thuật
- safety_critic: Kiểm tra an toàn và rủi ro
- executor: Tổng hợp chẩn đoán cuối cùng + thứ tự sửa chữa

Quy tắc:
- Nếu triệu chứng có dấu hiệu nguy hiểm (mùi gas, khói, điện giật) → ưu tiên safety_critic trước
- Nếu cần phân tích sâu triệu chứng → diagnostician
- Khi đã có đủ thông tin an toàn và chẩn đoán → executor
- Nếu đã có kết quả cuối cùng → next = "__end__"

Hãy suy nghĩ logic và trả về route rõ ràng."""

    response = llm.with_structured_output(Route).invoke([
        ("system", system),
        ("user", f"Triệu chứng: {state.task}\nCurrent messages: {str(state.messages[-3:]) if state.messages else 'Start'}")
    ])

    return {
        "next": response.next,
        "messages": [("assistant", f"Supervisor: {response.reasoning}")]
    }