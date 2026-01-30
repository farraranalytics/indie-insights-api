"""
Farrar Analytics - Indie Artist Insights API
Main FastAPI application
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
from typing import Optional

from .analytics_engine import DistroKidAnalyzer

app = FastAPI(
    title="Farrar Analytics API",
    description="Analytics engine for indie music artists",
    version="1.0.0"
)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:4321,https://farraranalytics.com").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database availability flag
DB_ENABLED = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_SERVICE_KEY"))


def get_db():
    """Lazy import database module only when needed"""
    from . import database
    return database


@app.get("/")
async def root():
    return {"message": "Farrar Analytics API", "status": "running"}


@app.get("/health")
async def health_check():
    has_url = bool(os.getenv("SUPABASE_URL"))
    has_key = bool(os.getenv("SUPABASE_SERVICE_KEY"))
    return {
        "status": "healthy",
        "db_enabled": DB_ENABLED,
        "has_url": has_url,
        "has_key": has_key,
        "url_len": len(os.getenv("SUPABASE_URL", "")),
        "key_len": len(os.getenv("SUPABASE_SERVICE_KEY", "")),
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: Optional[str] = None
):
    """
    Upload a DistroKid 'Excruciating Detail' file (TSV or XLSX)
    Process it and return analytics. Saves to DB if user_id provided.
    """

    if not file.filename.endswith(('.tsv', '.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a TSV, XLSX, or CSV file."
        )

    try:
        content = await file.read()

        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.tsv'):
            df = pd.read_csv(io.BytesIO(content), sep='\t')
        else:
            df = pd.read_csv(io.BytesIO(content))

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

        # Extract date range
        date_range_start = None
        date_range_end = None
        if 'Sale Month' in df.columns:
            months = pd.to_datetime(df['Sale Month'], errors='coerce').dropna()
            if len(months) > 0:
                date_range_start = str(months.min().date())
                date_range_end = str(months.max().date())

        # Add metadata
        results['metadata'] = {
            'filename': file.filename,
            'rows_processed': len(df),
            'user_id': user_id,
            'date_range_start': date_range_start,
            'date_range_end': date_range_end,
        }

        # Save to database if user_id provided and DB is configured
        saved_id = None
        if user_id and DB_ENABLED:
            try:
                db = get_db()
                saved = db.save_analysis(
                    user_id=user_id,
                    filename=file.filename,
                    row_count=len(df),
                    date_range_start=date_range_start,
                    date_range_end=date_range_end,
                    result=results,
                )
                saved_id = saved.get("id")
            except Exception as db_err:
                # Log but don't fail the request
                print(f"Warning: Failed to save analysis to DB: {db_err}")

        results['metadata']['saved_id'] = saved_id

        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/analyses/{user_id}")
async def get_analyses(user_id: str):
    """Get list of saved analyses for a user (metadata only, no full results)"""
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        db = get_db()
        analyses = db.get_user_analyses(user_id)
        return JSONResponse(content={"analyses": analyses})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analyses: {str(e)}")


@app.get("/analyses/{user_id}/{analysis_id}")
async def get_analysis(user_id: str, analysis_id: str):
    """Get a single saved analysis with full results"""
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        db = get_db()
        analysis = db.get_analysis_by_id(analysis_id, user_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return JSONResponse(content=analysis)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analysis: {str(e)}")


@app.delete("/analyses/{user_id}/{analysis_id}")
async def delete_analysis(user_id: str, analysis_id: str):
    """Delete a saved analysis"""
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        db = get_db()
        deleted = db.delete_analysis(analysis_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting analysis: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
