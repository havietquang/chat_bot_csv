import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
# =====================
# 1. CẤU HÌNH
# =====================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CSV_FILE = "sales.csv"

# =====================
# 2. HÀM XỬ LÝ CSV
# =====================
def add_row_to_csv(order_id, product, category, price, quantity, date):
    new_row = {
        "order_id": order_id,
        "product": product,
        "category": category,
        "price": price,
        "quantity": quantity,
        "date": date
    }

    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    return "✅ Đã thêm đơn hàng mới vào file CSV."

def calculate_sales(metric, group_by=None):
    df = pd.read_csv(CSV_FILE)
    df["revenue"] = df["price"] * df["quantity"]

    if metric == "total_revenue":
        result = (
            df.groupby(group_by)["revenue"].sum()
            if group_by else df["revenue"].sum()
        )

    elif metric == "total_quantity":
        result = (
            df.groupby(group_by)["quantity"].sum()
            if group_by else df["quantity"].sum()
        )

    elif metric == "average_price":
        result = (
            df.groupby(group_by)["price"].mean()
            if group_by else df["price"].mean()
        )

    else:
        return "❌ Metric không hợp lệ"

    # 👉 XỬ LÝ KIỂU DỮ LIỆU TRẢ VỀ
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return result.to_string()
    elif isinstance(result, (int, float, np.number)):
        return f"{result}"
    else:
        return str(result)

# =====================
# 3. ĐỌC CSV ĐỂ LÀM PROMPT
# =====================
try:
    df = pd.read_csv(CSV_FILE)
    full_data_string = df.to_string(index=False)
except Exception as e:
    print(f"❌ Lỗi đọc file CSV: {e}")
    exit()

# =====================
# 4. SYSTEM INSTRUCTION
# =====================
instruction = f"""
Bạn là AI phân tích dữ liệu CSV.

Dữ liệu từ file sales.csv:
-----------------
{full_data_string}
-----------------

QUY TẮC:
- Chỉ dùng dữ liệu trên
- Không bịa số
- Nếu người dùng yêu cầu THÊM dữ liệu → gọi add_row_to_csv
- Nếu người dùng yêu cầu TÍNH TOÁN (tổng, trung bình, thống kê) → gọi calculate_sales
- Nếu chỉ hỏi thông tin → trả lời bằng text
"""

# =====================
# 5. KHAI BÁO TOOL (FUNCTION CALLING)
# =====================
tools = [
    {
        "function_declarations": [
            {
                "name": "add_row_to_csv",
                "description": "Thêm một đơn hàng mới vào file sales.csv",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "integer"},
                        "product": {"type": "string"},
                        "category": {"type": "string"},
                        "price": {"type": "number"},
                        "quantity": {"type": "integer"},
                        "date": {"type": "string"}
                    },
                    "required": ["order_id","product","category","price","quantity","date"]
                }
            },
            {
                "name": "calculate_sales",
                "description": "Tính toán thống kê từ dữ liệu bán hàng",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": ["total_revenue", "total_quantity", "average_price"]
                        },
                        "group_by": {
                            "type": "string",
                            "enum": ["category", "product"]
                        }
                    },
                    "required": ["metric"]
                }
            }
        ]
    }
]


# =====================
# 6. KHỞI TẠO MODEL
# =====================
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=instruction,
    tools=tools
)

chat = model.start_chat(history=[])

print("🤖 Chatbot CSV Gemini sẵn sàng!")
print("👉 Gõ 'exit' để thoát")

# =====================
# 7. VÒNG LẶP CHAT
# =====================
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit", "thoát"]:
        break

    response = chat.send_message(user_input)
    part = response.candidates[0].content.parts[0]

    if hasattr(part, "function_call") and part.function_call:
        fc = part.function_call

        if fc.name == "add_row_to_csv":
            result = add_row_to_csv(**fc.args)
            print("\n🤖 Bot:", result)

        elif fc.name == "calculate_sales":
            result = calculate_sales(**fc.args)
            print("\n🤖 Bot (kết quả tính toán):\n", result)

    else:
        print("\n🤖 Bot:", part.text)
