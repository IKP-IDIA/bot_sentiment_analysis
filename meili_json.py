from fastapi import FastAPI
import json
import requests
import os
 
app = FastAPI()
 
# 🔹 URL ของ Meilisearch
MEILI_URL = "http://10.1.0.150:7700/indexes/web_scraping/documents"
 
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer MASTER_KEY"
}
 
# 🔹 Path ของไฟล์ JSON ที่ต้องการส่ง
JSON_FILE_PATH = r"C:\Users\artit\IKP_2025\Web_Scraping\crawl_output.json"
 
 
def read_and_send_json():
    """
    อ่านไฟล์ JSON, ส่งเข้า Meilisearch, และคืนข้อมูล JSON
    """
    if not os.path.exists(JSON_FILE_PATH):
        return {"status": "error", "message": f"ไม่พบไฟล์: {JSON_FILE_PATH}"}
 
    # อ่านข้อมูลจากไฟล์
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    if isinstance(data, dict):
        data_list = [data]
    else:
        data_list = data
 
    # ส่งเข้า Meilisearch
    try:
        response = requests.post(MEILI_URL, headers=HEADERS, json=data_list)
        if response.status_code == 202:
            status = "success"
            message = f"ส่งไฟล์ {os.path.basename(JSON_FILE_PATH)} สำเร็จ!"
        else:
            status = "error"
            message = f"ส่งไม่สำเร็จ ({response.status_code})"
    except Exception as e:
        status = "error"
        message = str(e)
 
    return {
        "status": status,
        "message": message,
        "json_content": data  # 🔹 แสดงเนื้อไฟล์ JSON
    }
 
 
# @app.post("/send-json")
# def send_json_post():
#     return read_and_send_json()
 
 
# @app.get("/send-json")
# def send_json_get():
#     return read_and_send_json()
 
 
 
# @app.get("/")
# def root():
#     return {"message": "FastAPI is running. ใช้ /send-json เพื่อส่ง JSON เข้า Meilisearch"}