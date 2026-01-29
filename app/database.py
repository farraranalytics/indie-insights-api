"""
Farrar Analytics - Database Connection
Simplified persistence: stores full analysis JSON per upload
"""

import os
from typing import Optional, Dict, Any, List
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Service role key for backend


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
    """Save a full analysis result to the database"""
    client = get_supabase_client()
    data = {
        "user_id": user_id,
        "filename": filename,
        "row_count": row_count,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "result": result,
    }
    response = client.table("analyses").insert(data).execute()
    return response.data[0] if response.data else {}


def get_user_analyses(user_id: str) -> List[Dict[str, Any]]:
    """Get all saved analyses for a user, newest first"""
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
    """Get a single analysis by ID (with user check)"""
    client = get_supabase_client()
    response = (
        client.table("analyses")
        .select("*")
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    return response.data[0] if response.data else None


def delete_analysis(analysis_id: str, user_id: str) -> bool:
    """Delete an analysis (with user check)"""
    client = get_supabase_client()
    response = (
        client.table("analyses")
        .delete()
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    return len(response.data) > 0 if response.data else False
