from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import logging
import uvicorn

# 1. ตั้งค่า Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for processing keywords and receiving aggregate sentiment results."
)

# 2. Pydantic Model สำหรับรับผลลัพธ์รวม (Micro-Payload)
class SentimentData(BaseModel):
    """
    Schema สำหรับผลลัพธ์รวมที่ Client Script จะส่งมาหลังจากวิเคราะห์เสร็จแล้ว
    """
    analysis_id: str = Field(..., description="Unique ID for this analysis run.")
    analysis_date: datetime
    keyword: str
    total_articles: int
    # ค่าเฉลี่ยของคะแนน Sentiment (ต้องอยู่ระหว่าง -1.0 ถึง 1.0)
    average_sentiment: float = Field(..., ge=-1.0, le=1.0) 
    # Label รวม: Positive, Neutral, หรือ Negative
    overall_label: str

# 3. Pydantic Model สำหรับรับ Keyword ใหม่ (จาก Client Script)
class KeywordInput(BaseModel):
    """
    Schema สำหรับรับ Keyword จาก Client เพื่อจำลองการเริ่มงาน
    """
    keyword: str = Field(..., description="The search keyword provided by the user.")

# --- API Endpoints ---

# 📌 NEW: Endpoint สำหรับรับ Keyword จาก Client Script (แทนที่ /extract)
@app.post("/analyze_keyword")
async def analyze_keyword(data: KeywordInput):
    """
    รับ Keyword จาก Client เพื่อจำลองการเริ่มกระบวนการวิเคราะห์ Sentiment.
    """
    logger.info(f"--- 🔑 KEYWORD RECEIVED ---")
    logger.info(f"Received Keyword: {data.keyword}")
    
    # 📌 ในการใช้งานจริง:
    # 1. Server จะเรียกใช้ฟังก์ชัน get_google_news(data.keyword)
    # 2. ทำการวิเคราะห์ sentiment
    # 3. บันทึกผลลัพธ์ หรือส่งผลลัพธ์กลับไป

    # สำหรับการสาธิต: แค่ตอบกลับด้วยข้อมูลที่รับมา
    return {
        "status": "processing_started",
        "message": f"Keyword '{data.keyword}' received and analysis initiated.",
        "keyword_received": data.keyword,
        "processed_at": datetime.now().isoformat()
    }


# 📌 EXISTING: Endpoint สำหรับรับผลลัพธ์รวม (Micro-Payload) จาก Python Script
@app.post("/api/sentiment")
async def receive_sentiment_data(data: SentimentData):
    """
    รับผลลัพธ์การวิเคราะห์ Sentiment (Micro-Payload) จาก Client
    """
    logger.info(f"--- 📬 AGGREGATE DATA RECEIVED ---")
    logger.info(f"Keyword: {data.keyword}")
    logger.info(f"Avg Sentiment: {data.average_sentiment:.4f} ({data.overall_label})")
    
    # ในการใช้งานจริง: บันทึก data ลงในฐานข้อมูล
    
    return {
        "status": "success",
        "message": "Aggregate Sentiment data processed and accepted.",
        "analysis_id": data.analysis_id,
        "average_sentiment": data.average_sentiment,
        "processed_at": datetime.now().isoformat()
    }

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running. Check /docs for endpoints."}

# --- 4. วิธีการรัน Server ---

if __name__ == "__main__":
    # ใช้ Uvicorn เพื่อรัน Server บน Localhost ที่ Port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
