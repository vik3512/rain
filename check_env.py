import os
from dotenv import load_dotenv

print("--- 正在檢查 .env 檔案讀取狀態 ---")

# 1. 檢查 .env 檔案是否存在並載入
#    load_dotenv() 會自動尋找當前目錄下的 .env 檔案
load_success = load_dotenv()
print(f"1. 尋找並載入 .env 檔案: {load_success}")

if not load_success:
    print("   [錯誤] 在目前資料夾中找不到 .env 檔案！")

# 2. 讀取 Google Key
gmap_key = os.getenv("GOOGLE_MAPS_API_KEY")

# 3. 讀取 OWM Key
owm_key = os.getenv("OPENWEATHER_API_KEY")

print("--- 檢查完畢 ---")

# 顯示 Google Key 檢查結果
if not gmap_key:
    print("\n[Google Key 檢查失敗!]")
    print("Python 讀不到 GOOGLE_MAPS_API_KEY。")
    print("請立刻檢查您的 .env 檔案：")
    print("  - 檢查 1 (檔名): 檔案名稱是否『完全』等於 .env (前面有點，後面沒有 .txt)")
    print("  - 檢查 2 (位置): .env 檔案是否和這個 check_env.py 在同一個資料夾 (RainApp/ 底下)")
    print("  - 檢查 3 (Key 名稱): .env 檔案裡的文字是否『完全』等於 GOOGLE_MAPS_API_KEY=... (包含底線)")
else:
    print(f"\n[Google Key 檢查成功!] 程式已讀取。 (您的 Key: {gmap_key[:4]}...{gmap_key[-4:]})")

# 顯示 OWM Key 檢查結果
if not owm_key:
    print("\n[OWM Key 檢查失敗!]")
    print("Python 讀不到 OPENWEATHER_API_KEY。")
    print("  (請依照上述三點檢查 .env 檔案)")
else:
    print(f"\n[OWM Key 檢查成功!] 程式已讀取。 (您的 Key: {owm_key[:4]}...{owm_key[-4:]})")