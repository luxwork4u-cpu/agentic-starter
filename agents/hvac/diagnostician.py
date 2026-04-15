from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def diagnostician_node(state: AgentState):
    # Lấy task an toàn từ dict
    task = state.get("task") if isinstance(state, dict) else getattr(state, "task", "")

    prompt = f"""Bạn là Diagnostician HVAC có hơn 15 năm kinh nghiệm thực tế tại Mỹ.
Bạn chuyên chẩn đoán chính xác và nhanh các vấn đề hệ thống điều hòa.

Triệu chứng: {task}

Hãy phân tích CHI TIẾT và theo thứ tự xác suất cao nhất → thấp hơn:
- Liệt kê 4-6 nguyên nhân phổ biến nhất
- Với mỗi nguyên nhân: giải thích rõ cơ chế tại sao nó gây ra triệu chứng (đặc biệt với trường hợp coil đóng băng)
- Đưa dấu hiệu phân biệt giữa các nguyên nhân
- Gợi ý cách kiểm tra nhanh (quick check) cho từng nguyên nhân

Ưu tiên các nguyên nhân thực tế phổ biến nhất với triệu chứng "coil frozen / evaporator frozen":
1. Low refrigerant (rò rỉ gas) - thường là nguyên nhân số 1
2. Low airflow (bộ lọc bẩn, quạt yếu, duct tắc)
3. Dirty evaporator coil
4. Bad expansion valve / TXV
5. Low ambient temperature + oversizing
6. Other (sensor, board, etc.)

Trả lời bằng tiếng Việt, ngắn gọn nhưng chuyên sâu, dễ hiểu cho technician."""

    result = llm.invoke([
        ("system", prompt),
        ("user", "Phân tích nguyên nhân theo thứ tự xác suất cao nhất và giải thích chi tiết.")
    ])

    content = result.content if hasattr(result, 'content') else str(result)

    return {
        "messages": [("assistant", f"Diagnostician: {content}")],
        "next": "supervisor"
    }