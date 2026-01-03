import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Cấu hình
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 2. Đọc file CSV
try:
    df = pd.read_csv("sales.csv")
    full_data_string = df.to_string(index=False)
except Exception as e:
    print(f"❌ Lỗi đọc file: {e}")
    exit()

# 3. SYSTEM INSTRUCTION (QUAN TRỌNG)
instruction = f"""
Bạn là một AI đọc và phân tích dữ liệu CSV.

Dưới đây là TOÀN BỘ dữ liệu từ file sales.csv:
-----------------
{full_data_string}
-----------------

QUY TẮC BẮT BUỘC:
- Chỉ sử dụng dữ liệu đã cung cấp
- KHÔNG suy đoán, KHÔNG bịa
- Nếu câu hỏi yêu cầu tổng / trung bình → ước lượng dựa trên dữ liệu
- Nếu câu hỏi yêu cầu chi tiết đơn hàng → trích đúng dòng
- Trả lời rõ ràng, có giải thích
"""

# 4. Khởi tạo model (KHÔNG tool)
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=instruction
)

chat = model.start_chat(history=[])

print("🤖 Chatbot đọc CSV sẵn sàng! (gõ 'exit' để thoát)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit", "thoát"]:
        break

    try:
        response = chat.send_message(user_input)
        print("\nBot:", response.text)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
