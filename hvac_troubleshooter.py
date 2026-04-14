from langchain_google_genai import ChatGoogleGenerativeAI
import os
from datetime import datetime

print("🔧 HVAC Troubleshooting Agent (Powered by Grok + Gemini)")
print("="*70)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def hvac_troubleshoot(symptoms: str):
    system_prompt = f"""Bạn là một HVAC Senior Technician có 15+ năm kinh nghiệm tại Mỹ.
Bạn chuyên chẩn đoán và sửa chữa hệ thống AC, Furnace, Heat Pump, Boiler, AHU, Fan Coil...

Quy tắc quan trọng:
- Luôn ưu tiên an toàn: nhắc kiểm tra điện, gas, refrigerant trước khi tháo.
- Đưa ra nguyên nhân theo thứ tự khả năng cao nhất → thấp hơn.
- Đưa thứ tự kiểm tra logic (từ dễ → khó, từ rẻ → đắt).
- Nếu triệu chứng nguy hiểm (mùi gas, khói, tiếng kêu lớn) → khuyên gọi pro ngay.
- Dùng ngôn ngữ đơn giản, dễ hiểu cho technician.

Triệu chứng từ khách hàng: {symptoms}

Hãy phân tích và trả lời theo cấu trúc rõ ràng:
1. **Tóm tắt vấn đề**
2. **Nguyên nhân có thể** (xếp theo xác suất cao → thấp)
3. **Thứ tự kiểm tra khuyến nghị** (bước 1, bước 2...)
4. **Tool cần chuẩn bị**
5. **Khi nào nên gọi thợ chuyên nghiệp**
6. **Phòng ngừa tương lai**

Bắt đầu phân tích ngay."""

    response = llm.invoke([
        ("system", system_prompt),
        ("user", "Phân tích và đưa ra hướng dẫn chẩn đoán chi tiết.")
    ])

    print(response.content)

# ================== CHẠY AGENT ==================
if __name__ == "__main__":
    print("Mô tả triệu chứng HVAC của bạn (bằng tiếng Việt hoặc English):")
    symptoms = input("→ ")
    
    if symptoms.strip():
        print("\n🔍 Đang chẩn đoán...\n")
        hvac_troubleshoot(symptoms)
    else:
        print("Vui lòng nhập triệu chứng!")