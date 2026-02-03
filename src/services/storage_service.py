"""
Storage Service for Supabase Storage.

Handles uploading documentation files to Supabase Storage bucket.
"""

import os
from typing import Optional
from supabase import create_client, Client
from ..config import settings


class StorageService:
    """
    Service for uploading documentation files to Supabase Storage.
    
    Files are stored in the 'documentation' bucket with structure:
    documentation/{job_id}/docs/...
    """
    
    BUCKET_NAME = "documentation"
    
    def __init__(self):
        """Initialize Supabase client with service role key."""
        if not settings.supabase_url or not settings.supabase_service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_key
        )
        self.storage = self.client.storage
    
    def upload_documentation(
        self, 
        job_id: str, 
        docs: dict[str, str]
    ) -> tuple[str, list[dict]]:
        """
        Upload all documentation files to Supabase Storage.
        
        Args:
            job_id: The job ID to use as folder name
            docs: Dictionary of {file_path: content}
            
        Returns:
            Tuple of (storage_base_path, list of file metadata)
            
        Raises:
            Exception: If upload fails
        """
        storage_base_path = f"{job_id}/"
        uploaded_files = []
        
        for file_path, content in docs.items():
            # Full path in storage: {job_id}/docs/charts/00_INDEX.md
            full_path = f"{storage_base_path}{file_path}"
            
            # Convert content to bytes
            content_bytes = content.encode('utf-8')
            
            try:
                # Upload file to storage
                result = self.storage.from_(self.BUCKET_NAME).upload(
                    path=full_path,
                    file=content_bytes,
                    file_options={
                        "content-type": "text/markdown",
                        "upsert": "true"  # Overwrite if exists
                    }
                )
                
                # Get file metadata
                file_meta = {
                    "path": file_path,
                    "storage_path": full_path,
                    "size": len(content_bytes),
                    "content_type": "text/markdown"
                }
                uploaded_files.append(file_meta)
                
                print(f"[StorageService] Uploaded: {full_path}")
                
            except Exception as e:
                print(f"[StorageService] Error uploading {full_path}: {e}")
                # Try to upsert (update if exists)
                try:
                    self.storage.from_(self.BUCKET_NAME).update(
                        path=full_path,
                        file=content_bytes,
                        file_options={"content-type": "text/markdown"}
                    )
                    file_meta = {
                        "path": file_path,
                        "storage_path": full_path,
                        "size": len(content_bytes),
                        "content_type": "text/markdown"
                    }
                    uploaded_files.append(file_meta)
                    print(f"[StorageService] Updated: {full_path}")
                except Exception as e2:
                    print(f"[StorageService] Failed to upload/update {full_path}: {e2}")
                    raise
        
        return storage_base_path, uploaded_files
    
    def get_public_url(self, storage_path: str) -> str:
        """
        Get the public URL for a file in storage.
        
        Args:
            storage_path: Full path in storage (e.g., job_id/docs/STRUCTURE.md)
            
        Returns:
            Public URL string
        """
        result = self.storage.from_(self.BUCKET_NAME).get_public_url(storage_path)
        return result
    
    def get_file_content(self, storage_path: str) -> Optional[str]:
        """
        Download and return file content.
        
        Args:
            storage_path: Full path in storage
            
        Returns:
            File content as string, or None if not found
        """
        try:
            result = self.storage.from_(self.BUCKET_NAME).download(storage_path)
            return result.decode('utf-8')
        except Exception as e:
            print(f"[StorageService] Error downloading {storage_path}: {e}")
            return None
    
    def list_files(self, job_id: str) -> list[dict]:
        """
        List all documentation files for a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            List of file metadata objects
        """
        try:
            storage_path = f"{job_id}/"
            result = self.storage.from_(self.BUCKET_NAME).list(storage_path)
            return result
        except Exception as e:
            print(f"[StorageService] Error listing files for {job_id}: {e}")
            return []
    
    def delete_job_files(self, job_id: str) -> bool:
        """
        Delete all documentation files for a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # List all files first
            files = self.list_files(job_id)
            if not files:
                return True
            
            # Build list of paths to delete
            paths = [f"{job_id}/{f['name']}" for f in files]
            
            # Delete files
            self.storage.from_(self.BUCKET_NAME).remove(paths)
            print(f"[StorageService] Deleted {len(paths)} files for job {job_id}")
            return True
            
        except Exception as e:
            print(f"[StorageService] Error deleting files for {job_id}: {e}")
            return False


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create the storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
