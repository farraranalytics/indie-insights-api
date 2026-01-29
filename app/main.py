"""
Farrar Analytics - Indie Artist Insights API
Main FastAPI application
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import Optional

from .analytics_engine import DistroKidAnalyzer
from .database import get_supabase_client, save_upload, save_streams_data, get_user_data

app = FastAPI(
    title="Farrar Analytics API",
    description="Analytics engine for indie music artists",
    version="1.0.0"
)

# CORS - update origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://farraranalytics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Farrar Analytics API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: Optional[str] = None  # Will come from auth in production
):
    """
    Upload a DistroKid 'Excruciating Detail' file (TSV or XLSX)
    Process it and return analytics
    """
    
    # Validate file type
    if not file.filename.endswith(('.tsv', '.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a TSV, XLSX, or CSV file."
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse based on file type
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.tsv'):
            df = pd.read_csv(io.BytesIO(content), sep='\t')
        else:  # CSV
            df = pd.read_csv(io.BytesIO(content))
        
        # Validate required columns
        required_columns = ['Sale Month', 'Store', 'Title', 'Quantity', 'Earnings (USD)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}. Please upload a DistroKid 'Excruciating Detail' export."
            )
        
        # Run analytics
        analyzer = DistroKidAnalyzer(df)
        results = analyzer.get_full_analysis()
        
        # Add metadata
        results['metadata'] = {
            'filename': file.filename,
            'rows_processed': len(df),
            'user_id': user_id
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str):
    """
    Get cached dashboard data for a user
    """
    # TODO: Implement database lookup
    return {"message": "Dashboard endpoint", "user_id": user_id}


@app.get("/dashboard/{user_id}/overview")
async def get_overview(user_id: str):
    """Get overview metrics for user"""
    # TODO: Implement
    pass


@app.get("/dashboard/{user_id}/songs")
async def get_songs(user_id: str):
    """Get song breakdown for user"""
    # TODO: Implement
    pass


@app.get("/dashboard/{user_id}/platforms")
async def get_platforms(user_id: str):
    """Get platform breakdown for user"""
    # TODO: Implement
    pass


@app.get("/dashboard/{user_id}/countries")
async def get_countries(user_id: str):
    """Get country breakdown for user"""
    # TODO: Implement
    pass


@app.get("/dashboard/{user_id}/trends")
async def get_trends(user_id: str):
    """Get monthly trends for user"""
    # TODO: Implement
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
