import os
from supabase import create_client, Client
from dotenv import load_dotenv
import json
from typing import Optional, List, Dict, Any

# Load environment variables
load_dotenv()

# Defaults for local development/fallback to prevent crash on import
url: str = os.environ.get("SUPABASE_URL", "")
key: str = os.environ.get("SUPABASE_KEY", "")

# Warning if not configured
if not url or not key:
    print("⚠️ WARNING: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")
    print("   Database operations will fail until configured.")

try:
    if url and key:
        supabase: Client = create_client(url, key)
    else:
        # Create a dummy client object or None, but simpler to just not define it 
        # and handle usage errors? Or use a Mock?
        # Let's try to initialize with empty strings? No that failed.
        # We'll define it as None and check in methods.
        supabase = None
except Exception as e:
    print(f"❌ Error initializing Supabase client: {e}")
    supabase = None

class Database:
    """Wrapper class for Supabase database operations."""
    
    @staticmethod
    def upsert_paper(paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert or update a paper record."""
        if not supabase:
            return {}
            
        # Ensure authors is a valid JSON field
        if "authors" in paper_data and isinstance(paper_data["authors"], list):
            # authors is already a list, which is valid for jsonb in python client
            pass
            
        try:
            # We use upsert based on DOI or ID if present
            # For now, let's assume we want to return the inserted data
            response = supabase.table("papers").upsert(paper_data).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            print(f"❌ Error upserting paper: {e}")
            return {}

    @staticmethod
    def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
        """Get a paper by UUID."""
        if not supabase:
            return None
        try:
            response = supabase.table("papers").select("*").eq("id", paper_id).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            print(f"❌ Error fetching paper: {e}")
            return None

    @staticmethod
    def get_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
        """Get a paper by DOI."""
        if not supabase:
            return None
        try:
            response = supabase.table("papers").select("*").eq("doi", doi).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            print(f"❌ Error fetching paper by DOI: {e}")
            return None

    @staticmethod
    def upsert_claim(claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert or update a claim."""
        if not supabase:
            return {}
        try:
            response = supabase.table("claims").upsert(claim_data).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            print(f"❌ Error upserting claim: {e}")
            return {}
            
    @staticmethod
    def get_claims_for_paper(paper_id: str) -> List[Dict[str, Any]]:
        """Get all claims for a specific paper."""
        if not supabase:
            return []
        try:
            response = supabase.table("claims").select("*").eq("paper_id", paper_id).execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"❌ Error fetching claims: {e}")
            return []

    # --- Conversation & Messages ---

    @staticmethod
    def create_conversation(title: str = "New Research Session") -> Dict[str, Any]:
        """Create a new conversation."""
        if not supabase:
            return {}
        try:
            data = {"title": title}
            response = supabase.table("conversations").insert(data).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            print(f"❌ Error creating conversation: {e}")
            return {}

    @staticmethod
    def add_message(conversation_id: str, role: str, content: Any) -> Dict[str, Any]:
        """Add a message to a conversation. Content can be text or dict (JSON)."""
        if not supabase:
            return {}
        try:
            # Ensure content is JSON serializable if needed, but the client handles dict->jsonb
            data = {
                "conversation_id": conversation_id,
                "role": role,
                "content": content
            }
            response = supabase.table("messages").insert(data).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            print(f"❌ Error adding message: {e}")
            return {}

    @staticmethod
    def get_conversation_history(conversation_id: str) -> List[Dict[str, Any]]:
        """Get full history for a conversation."""
        if not supabase:
            return []
        try:
            response = supabase.table("messages")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at")\
                .execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"❌ Error fetching history: {e}")
            return []

    # --- System ---

    @staticmethod
    def health_check() -> bool:
        """Check if database is reachable."""
        if not supabase:
            return False
        try:
            # Simple query to check connection
            supabase.table("papers").select("count", count="exact").limit(1).execute()
            return True
        except Exception as e:
            # print(f"❌ DB Health Check failed: {e}")
            return False

    @staticmethod
    def clear_all() -> bool:
        """Clear all data from tables (for demo/reset)."""
        if not supabase:
            return False
        try:
            # Delete in order of dependencies (child first)
            # Note: Supabase/PostgREST delete requires a filter. neq id 0 is a hack to delete all.
            fake_uuid = "00000000-0000-0000-0000-000000000000"
            
            supabase.table("citations").delete().neq("id", fake_uuid).execute()
            supabase.table("messages").delete().neq("id", fake_uuid).execute()
            
            # Claims depends on papers, Messages depends on Conversations
            supabase.table("citations").delete().neq("id", fake_uuid).execute() 
            # Re-running citations just in case, but actually messages -> conversations
            
            supabase.table("claims").delete().neq("id", fake_uuid).execute()
            supabase.table("papers").delete().neq("id", fake_uuid).execute()
            supabase.table("conversations").delete().neq("id", fake_uuid).execute()
            return True
        except Exception as e:
            print(f"❌ Error clearing DB: {e}")
            return False

    @staticmethod
    def get_claim_by_id(claim_id: str) -> Optional[Dict[str, Any]]:
        """Get a claim by ID."""
        if not supabase:
            return None
        try:
            response = supabase.table("claims").select("*").eq("id", claim_id).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            print(f"❌ Error fetching claim: {e}")
            return None
