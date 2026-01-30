"""
Farrar Analytics - Indie Artist Insights API
Main FastAPI application
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import logging
import asyncio
import math
from functools import partial
from typing import Optional

from .analytics_engine import DistroKidAnalyzer

logger = logging.getLogger("indie-insights")
logging.basicConfig(level=logging.INFO)


def sanitize_for_json(obj):
    """Replace NaN/Infinity floats with None so the response is valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    return obj

# 50 MB file size limit
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024

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
    return {"status": "healthy", "db_enabled": DB_ENABLED}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
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
        file_size = len(content)
        logger.info(f"Upload received: {file.filename} ({file_size / 1024:.1f} KB)")

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum size is {MAX_FILE_SIZE // 1024 // 1024} MB."
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Parse file in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = await loop.run_in_executor(None, partial(pd.read_excel, io.BytesIO(content)))
        elif file.filename.endswith('.tsv'):
            df = await loop.run_in_executor(None, partial(pd.read_csv, io.BytesIO(content), sep='\t'))
        else:
            df = await loop.run_in_executor(None, partial(pd.read_csv, io.BytesIO(content)))

        logger.info(f"Parsed {len(df)} rows from {file.filename}")

        required_columns = ['Sale Month', 'Store', 'Title', 'Quantity', 'Earnings (USD)']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}. Please upload a DistroKid 'Excruciating Detail' export."
            )

        # Run analytics in thread pool to avoid blocking the event loop
        def run_analysis(dataframe):
            analyzer = DistroKidAnalyzer(dataframe)
            return analyzer.get_full_analysis()

        results = await loop.run_in_executor(None, run_analysis, df)

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

        logger.info(f"Analysis complete for {file.filename}")

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
                logger.info(f"Saved analysis {saved_id} for user {user_id}")
            except Exception as db_err:
                logger.warning(f"Failed to save analysis to DB: {db_err}")

        results['metadata']['saved_id'] = saved_id

        return JSONResponse(content=sanitize_for_json(results))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
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
        return JSONResponse(content=sanitize_for_json(analysis))
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
