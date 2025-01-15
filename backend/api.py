# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from news_analyzer import NewsAnalyzer  # This is the class we created earlier

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our analyzer
analyzer = NewsAnalyzer()

# Pydantic models for request validation
class ArticleRequest(BaseModel):
    arabic_url: str
    western_url: str

class HistoricalRequest(BaseModel):
    urls: List[ArticleRequest]
    timeframe: Optional[str] = "1M"

@app.post("/api/analyze")
async def analyze_articles(request: ArticleRequest):
    try:
        analysis = analyzer.compare_articles(request.arabic_url, request.western_url)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/historical")
async def analyze_historical(request: HistoricalRequest):
    try:
        urls_list = [(article.arabic_url, article.western_url) for article in request.urls]
        analysis = analyzer.historical_analysis(urls_list, request.timeframe)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)