"""
Farrar Analytics - Database Connection
Supabase integration for user data, uploads, and cached analytics
"""

import os
from typing import Optional, Dict, Any
from supabase import create_client, Client
from datetime import datetime

# Get Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # Use anon/public key for client, service key for server


def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ==================== User Operations ====================

def create_user(email: str, artist_name: str) -> Dict[str, Any]:
    """Create a new user record"""
    client = get_supabase_client()
    
    data = {
        "email": email,
        "artist_name": artist_name,
        "created_at": datetime.utcnow().isoformat()
    }
    
    result = client.table("users").insert(data).execute()
    return result.data[0] if result.data else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID"""
    client = get_supabase_client()
    result = client.table("users").select("*").eq("id", user_id).execute()
    return result.data[0] if result.data else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    client = get_supabase_client()
    result = client.table("users").select("*").eq("email", email).execute()
    return result.data[0] if result.data else None


# ==================== Upload Operations ====================

def save_upload(
    user_id: str,
    filename: str,
    file_path: str,
    row_count: int,
    date_range_start: str,
    date_range_end: str
) -> Dict[str, Any]:
    """Save upload metadata"""
    client = get_supabase_client()
    
    data = {
        "user_id": user_id,
        "filename": filename,
        "file_path": file_path,
        "row_count": row_count,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    result = client.table("uploads").insert(data).execute()
    return result.data[0] if result.data else None


def get_user_uploads(user_id: str) -> list:
    """Get all uploads for a user"""
    client = get_supabase_client()
    result = client.table("uploads")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("uploaded_at", desc=True)\
        .execute()
    return result.data


# ==================== Stream Data Operations ====================

def save_streams_data(user_id: str, upload_id: str, streams_data: list) -> int:
    """
    Bulk insert stream data from upload
    Returns number of rows inserted
    """
    client = get_supabase_client()
    
    # Add user_id and upload_id to each record
    records = []
    for stream in streams_data:
        records.append({
            "user_id": user_id,
            "upload_id": upload_id,
            "sale_month": stream.get("Sale Month"),
            "store": stream.get("Store"),
            "title": stream.get("Title"),
            "isrc": stream.get("ISRC"),
            "country": stream.get("Country of Sale"),
            "quantity": stream.get("Quantity"),
            "earnings": stream.get("Earnings (USD)"),
            "created_at": datetime.utcnow().isoformat()
        })
    
    # Batch insert (Supabase handles chunking)
    if records:
        result = client.table("streams").insert(records).execute()
        return len(result.data) if result.data else 0
    return 0


def get_user_data(user_id: str) -> list:
    """Get all stream data for a user"""
    client = get_supabase_client()
    result = client.table("streams")\
        .select("*")\
        .eq("user_id", user_id)\
        .execute()
    return result.data


def get_user_data_by_date_range(user_id: str, start_date: str, end_date: str) -> list:
    """Get stream data for a user within date range"""
    client = get_supabase_client()
    result = client.table("streams")\
        .select("*")\
        .eq("user_id", user_id)\
        .gte("sale_month", start_date)\
        .lte("sale_month", end_date)\
        .execute()
    return result.data


# ==================== Cached Metrics Operations ====================

def save_monthly_metrics(user_id: str, metrics: list) -> int:
    """Save pre-aggregated monthly metrics"""
    client = get_supabase_client()
    
    records = []
    for m in metrics:
        records.append({
            "user_id": user_id,
            "sale_month": m.get("month"),
            "total_earnings": m.get("earnings"),
            "total_streams": m.get("streams"),
            "updated_at": datetime.utcnow().isoformat()
        })
    
    if records:
        # Upsert to handle updates
        result = client.table("monthly_metrics").upsert(records).execute()
        return len(result.data) if result.data else 0
    return 0


def get_monthly_metrics(user_id: str) -> list:
    """Get cached monthly metrics for a user"""
    client = get_supabase_client()
    result = client.table("monthly_metrics")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("sale_month", desc=False)\
        .execute()
    return result.data


# ==================== File Storage Operations ====================

def upload_file_to_storage(user_id: str, filename: str, file_content: bytes) -> str:
    """
    Upload file to Supabase storage
    Returns the file path
    """
    client = get_supabase_client()
    
    # Create unique path
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = f"{user_id}/{timestamp}_{filename}"
    
    # Upload to 'uploads' bucket
    result = client.storage.from_("uploads").upload(file_path, file_content)
    
    return file_path


def get_file_url(file_path: str) -> str:
    """Get public URL for a file"""
    client = get_supabase_client()
    result = client.storage.from_("uploads").get_public_url(file_path)
    return result
