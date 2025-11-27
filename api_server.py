from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import logging
import uvicorn # จำเป็นสำหรับการรัน Server

# 1. ตั้งค่า Logging (เพื่อให้เห็นข้อมูลเข้าใน Terminal)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Data Receiver API",
    description="API Endpoint for receiving aggregate sentiment analysis results from the Python script."
)

# 2. Pydantic Model สำหรับ Payload ที่แก้ไขแล้ว 

class SentimentData(BaseModel):
    """
    Schema สำหรับผลลัพธ์รวมเท่านั้น (Micro-Payload)
    """
    analysis_id: str = Field(..., description="Unique ID for this analysis run.")
    analysis_date: datetime
    keyword: str
    total_articles: int
    # ค่าเฉลี่ยของคะแนน Sentiment (ต้องอยู่ระหว่าง -1.0 ถึง 1.0)
    average_sentiment: float = Field(..., ge=-1.0, le=1.0) 
    # Label รวม: Positive, Neutral, หรือ Negative
    overall_label: str
    
    # หมายเหตุ: ไม่รวม news_articles ซึ่งเป็น list of dicts

# 3. API Endpoint

@app.post("/api/sentiment")
async def receive_sentiment_data(data: SentimentData):
    """
    รับผลลัพธ์การวิเคราะห์ Sentiment (Micro-Payload) จาก Client
    """
    logger.info(f"--- 📬 DATA RECEIVED ---")
    logger.info(f"Keyword: {data.keyword}")
    logger.info(f"Avg Sentiment: {data.average_sentiment:.4f} ({data.overall_label})")
    
    # 📌 ในการใช้งานจริง:
    # 1. เชื่อมต่อฐานข้อมูล (Database)
    # 2. บันทึกข้อมูล aggregate (data.dict()) ลงในฐานข้อมูล
    
    return {
        "status": "success",
        "message": "Aggregate Sentiment data processed and accepted.",
        "analysis_id": data.analysis_id,
        "average_sentiment": data.average_sentiment,
        "processed_at": datetime.now().isoformat()
    }

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running."}

# --- 4. วิธีการรัน Server ---

if __name__ == "__main__":
    # ใช้ Uvicorn เพื่อรัน Server
    # โฮสต์: 127.0.0.1 (Localhost), พอร์ต: 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)