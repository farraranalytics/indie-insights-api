"""
Farrar Analytics - Database Connection
Persistence: metadata in DB, full analysis JSON in Supabase Storage (gzipped)
"""

import os
import json
import gzip
import uuid
import logging
from typing import Optional, Dict, Any, List
from supabase import create_client, Client

logger = logging.getLogger("indie-insights")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Service role key for backend

STORAGE_BUCKET = "analysis-results"

_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create Supabase client (singleton)"""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def save_analysis(
    user_id: str,
    filename: str,
    row_count: int,
    date_range_start: Optional[str],
    date_range_end: Optional[str],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Save analysis: upload result JSON to Storage, save metadata to DB"""
    client = get_supabase_client()

    # Upload gzipped result JSON to Supabase Storage
    storage_path = f"{user_id}/{uuid.uuid4()}.json.gz"
    result_bytes = gzip.compress(json.dumps(result).encode("utf-8"))
    logger.info(f"Uploading analysis to storage: {storage_path} ({len(result_bytes) / 1024:.1f} KB compressed)")

    client.storage.from_(STORAGE_BUCKET).upload(
        storage_path,
        result_bytes,
        file_options={"content-type": "application/gzip"},
    )

    # Save metadata to DB (no large result blob)
    data = {
        "user_id": user_id,
        "filename": filename,
        "row_count": row_count,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "storage_path": storage_path,
    }
    response = client.table("analyses").insert(data).execute()
    return response.data[0] if response.data else {}


def get_user_analyses(user_id: str) -> List[Dict[str, Any]]:
    """Get all saved analyses for a user, newest first (metadata only)"""
    client = get_supabase_client()
    response = (
        client.table("analyses")
        .select("id, filename, row_count, date_range_start, date_range_end, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


def get_analysis_by_id(analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a single analysis by ID: metadata from DB, result from Storage"""
    client = get_supabase_client()
    response = (
        client.table("analyses")
        .select("*")
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not response.data:
        return None

    analysis = response.data[0]

    # Download result from Storage
    storage_path = analysis.get("storage_path")
    if storage_path:
        result_bytes = client.storage.from_(STORAGE_BUCKET).download(storage_path)
        result = json.loads(gzip.decompress(result_bytes).decode("utf-8"))
        analysis["result"] = result
        # Remove storage_path from response (frontend doesn't need it)
        analysis.pop("storage_path", None)

    return analysis


def delete_analysis(analysis_id: str, user_id: str) -> bool:
    """Delete an analysis: remove from DB and Storage"""
    client = get_supabase_client()

    # Get storage path before deleting
    response = (
        client.table("analyses")
        .select("storage_path")
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    storage_path = response.data[0].get("storage_path") if response.data else None

    # Delete from DB
    del_response = (
        client.table("analyses")
        .delete()
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    deleted = len(del_response.data) > 0 if del_response.data else False

    # Delete from Storage
    if deleted and storage_path:
        try:
            client.storage.from_(STORAGE_BUCKET).remove([storage_path])
        except Exception as e:
            logger.warning(f"Failed to delete storage file {storage_path}: {e}")

    return deleted
