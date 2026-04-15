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
    # Lấy task an toàn từ dict hoặc object
    task = state.get("task") if isinstance(state, dict) else getattr(state, "task", "")

    system = """Bạn là Supervisor của HVAC Troubleshooting Team.

Available routes: diagnostician, safety_critic, executor, __end__

Quy tắc:
- Nếu triệu chứng có dấu hiệu nguy hiểm (mùi gas, khói, điện) → safety_critic trước
- Nếu cần phân tích nguyên nhân → diagnostician
- Khi đã có đủ thông tin an toàn và chẩn đoán → executor
- Khi xong → __end__

Hãy suy nghĩ ngắn gọn và trả về route phù hợp."""

    response = llm.with_structured_output(Route).invoke([
        ("system", system),
        ("user", f"Triệu chứng: {task}\nCurrent messages: {str(state.get('messages', [])[-3:]) if isinstance(state, dict) else 'Start'}")
    ])

    return {
        "next": response.next,
        "messages": [("assistant", f"Supervisor: {response.reasoning}")]
    }