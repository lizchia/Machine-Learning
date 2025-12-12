import asyncio
import json
import pandas as pd
import re
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# 1. 嘗試載入 .env 檔案
# 如果載入成功，它會回傳 True；如果找不到檔案，回傳 False
if not load_dotenv():
    print("⚠️  警告：找不到 .env 檔案。")
    print("請確認您已將 .env.example 複製為 .env 並填入 API Key。")
    # 視情況決定要不要強制結束程式
    # sys.exit(1)

# 2. 讀取變數
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
base_url = os.getenv("BASE_URL")
debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"

# 3. 檢查關鍵變數是否存在
if not api_key:
    print("❌ 錯誤：未偵測到 OPENAI_API_KEY！無法執行後續程式。")
    sys.exit(1)

# 4. (測試用) 印出當前設定 (注意：不要印出完整的 api_key)
if debug_mode:
    print(f"✅ 環境載入成功")
    print(f"   - 使用模型: {model_name}")
    print(f"   - API Key: {api_key[:8]}********") # 只印前8碼檢查用
    print("-----------------------------------")

client = AsyncOpenAI(
    api_key = api_key,
    base_url = base_url
)

# 檔案路徑
ORIGINAL_CSV = "final_dataset.csv"
OUTPUT_FILE = "final_merged_dataset.csv"  # 最終產出的完整檔案
TEMP_FILE = "temp_synthetic_data.csv"     # 暫存檔

# 生成目標
TARGET_COUNT = 5 # 測試用，您可以改為 27344
CONCURRENT_LIMIT = 5
# ============================================

# 讀取原始 CSV 以獲取欄位結構
try:
    df_orig = pd.read_csv(ORIGINAL_CSV)
    ORIGINAL_COLUMNS = df_orig.columns.tolist()
    print(f"📖 原始資料欄位: {ORIGINAL_COLUMNS}")
except Exception as e:
    print(f"❌ 無法讀取原始 CSV: {e}")
    exit()

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: return None
        return None

async def generate_single_record(sem, pbar, buffer):
    async with sem:
        # Prompt 明確要求輸出對應欄位的 Key
        prompt = f"""
        You are a simulator for MIMIC-IV ICU patients.
        Generate a HYPOTHETICAL patient profile who will have a **Long ICU Stay (>7 days)**.

        Generate a JSON object with these EXACT keys:
        1. "text": Clinical note (Chief Complaint & HPI only).
        2. "age": Float (18.0 - 90.0).
        3. "gender_code": Integer (0 for Female, 1 for Male).
        4. "is_intubated": Integer (0 or 1).
        5. "min_map": Float (Lowest MAP in 24h, e.g., 40-70).
        6. "max_resp_rate": Float (Highest RR in 24h, e.g., 20-45).

        Output ONLY JSON.
        """
        
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=800
            )
            
            data = extract_json_from_text(response.choices[0].message.content.strip())
            
            if data:
                # ⬇️ 這裡就是「欄位對齊」的關鍵 ⬇️
                new_row = {
                    # 1. 填入 LLM 生成的數值 (對應原始欄位)
                    "text": data.get("text", ""),
                    "age": float(data.get("age", 60.0)),
                    "gender_code": int(data.get("gender_code", 1)),
                    "is_intubated": int(data.get("is_intubated", 0)),
                    "min_map": float(data.get("min_map", 65.0)),
                    "max_resp_rate": float(data.get("max_resp_rate", 20.0)),
                    
                    # 2. 填入目標標籤 (因為我們生成的是長住院)
                    "target_long_stay": 1, 
                    "los": 8.0, # 假定一個 >7 的數值
                    
                    # 3. 填入 ID 類欄位 (用 -999 填充，避免空值)
                    "subject_id": -999,
                    "hadm_id": -999,
                    "stay_id": -999,
                    "intime": "2100-01-01 00:00:00", # 假時間
                    
                    # 4. 標記這是合成資料 (方便之後辨識)
                    "is_synthetic": 1
                }
                
                # 確保新資料包含原始 CSV 的所有欄位 (如果沒填到的補 0)
                # 這樣 Concat 時才不會錯位
                final_row_cleaned = {}
                for col in ORIGINAL_COLUMNS:
                    final_row_cleaned[col] = new_row.get(col, 0) # 如果 new_row 沒這個欄位，就填 0
                
                # 補上我們額外加的 is_synthetic
                final_row_cleaned['is_synthetic'] = 1

                buffer.append(final_row_cleaned)
                pbar.update(1)

        except Exception as e:
            pass

async def main():
    print(f"🚀 開始生成並對齊欄位...")
    
    # 初始化暫存檔 (如果不存在，寫入 Header)
    # Header 必須包含原始欄位 + is_synthetic
    cols_to_save = ORIGINAL_COLUMNS + ['is_synthetic']
    if not os.path.exists(TEMP_FILE):
        pd.DataFrame(columns=cols_to_save).to_csv(TEMP_FILE, index=False)

    sem = asyncio.Semaphore(CONCURRENT_LIMIT)
    buffer = []
    
    tasks = []
    for _ in range(TARGET_COUNT):
        tasks.append(generate_single_record(sem, tqdm(total=TARGET_COUNT), buffer))
    
    # 簡單版：等待全部完成 (正式跑可用前面提供的分批寫法)
    await asyncio.gather(*tasks)
    
    if buffer:
        # 存入暫存
        df_buffer = pd.DataFrame(buffer)
        df_buffer.to_csv(TEMP_FILE, mode='a', header=False, index=False)

    # --- 最後合併 ---
    print("🔄 正在合併...")
    df_orig = pd.read_csv(ORIGINAL_CSV)
    df_orig['is_synthetic'] = 0 # 原始資料標記為 0
    
    df_synth = pd.read_csv(TEMP_FILE)
    
    # 這裡使用 pd.concat，因為欄位名稱完全一致，它會自動對齊上下
    df_final = pd.concat([df_orig, df_synth], ignore_index=True)
    
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ 完成！檔案結構已對齊。")
    print(f"預覽合成資料欄位:\n{df_final[df_final['is_synthetic']==1][['age', 'min_map', 'gender_code']].head(3)}")

if __name__ == "__main__":
    asyncio.run(main())