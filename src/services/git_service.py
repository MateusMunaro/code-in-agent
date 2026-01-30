"""
Git service for cloning and managing repositories.
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Optional

from git import Repo, GitCommandError


class GitService:
    """Service for Git operations."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_repo_path(self, job_id: str) -> Path:
        """Get the path where a repository should be cloned."""
        return self.base_path / job_id

    async def clone_repository(self, repo_url: str, job_id: str, branch: str = None) -> str:
        """
        Clone a repository asynchronously.
        
        Args:
            repo_url: The URL of the repository to clone
            job_id: The job ID (used as folder name)
            branch: Optional branch to checkout
            
        Returns:
            Path to the cloned repository
        """
        repo_path = self._get_repo_path(job_id)

        # Remove existing directory if present
        if repo_path.exists():
            shutil.rmtree(repo_path)

        # Clone in a thread pool to not block the event loop
        def _clone():
            clone_kwargs = {
                "url": repo_url,
                "to_path": str(repo_path),
                "depth": 1,  # Shallow clone for faster processing
            }
            
            if branch:
                clone_kwargs["branch"] = branch

            try:
                Repo.clone_from(**clone_kwargs)
            except GitCommandError as e:
                raise RuntimeError(f"Failed to clone repository: {e}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _clone)

        return str(repo_path)

    async def get_file_content(self, repo_path: str, file_path: str) -> Optional[str]:
        """
        Read the content of a file from the repository.
        
        Args:
            repo_path: Path to the repository
            file_path: Relative path to the file
            
        Returns:
            File content as string, or None if file doesn't exist
        """
        full_path = Path(repo_path) / file_path

        if not full_path.exists() or not full_path.is_file():
            return None

        def _read():
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read)

    async def list_files(
        self,
        repo_path: str,
        extensions: list[str] = None,
        exclude_patterns: list[str] = None
    ) -> list[str]:
        """
        List all files in the repository.
        
        Args:
            repo_path: Path to the repository
            extensions: Optional list of extensions to filter (e.g., ['.py', '.js'])
            exclude_patterns: Patterns to exclude (e.g., ['node_modules', '__pycache__'])
            
        Returns:
            List of relative file paths
        """
        default_excludes = [
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
            "target",
            ".idea",
            ".vscode",
            "vendor",
            "coverage",
            ".pytest_cache",
            ".mypy_cache",
        ]

        exclude_patterns = exclude_patterns or default_excludes
        repo_root = Path(repo_path)
        files = []

        def _list():
            for item in repo_root.rglob("*"):
                if not item.is_file():
                    continue

                # Check exclusions
                rel_path = str(item.relative_to(repo_root))
                if any(exc in rel_path for exc in exclude_patterns):
                    continue

                # Check extensions
                if extensions and item.suffix not in extensions:
                    continue

                files.append(rel_path)

            return files

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def get_repo_info(self, repo_path: str) -> dict:
        """
        Get basic information about the repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary with repo information
        """
        def _get_info():
            try:
                repo = Repo(repo_path)
                
                # Get remote URL
                remotes = [r.url for r in repo.remotes] if repo.remotes else []
                
                # Get current branch
                try:
                    current_branch = repo.active_branch.name
                except TypeError:
                    current_branch = "HEAD"
                
                # Get last commit
                try:
                    last_commit = repo.head.commit
                    commit_info = {
                        "sha": last_commit.hexsha[:8],
                        "message": last_commit.message.strip()[:100],
                        "author": str(last_commit.author),
                        "date": last_commit.committed_datetime.isoformat(),
                    }
                except Exception:
                    commit_info = None

                return {
                    "remotes": remotes,
                    "branch": current_branch,
                    "last_commit": commit_info,
                }
            except Exception as e:
                return {"error": str(e)}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_info)

    async def cleanup(self, repo_path: str):
        """
        Remove a cloned repository.
        
        Args:
            repo_path: Path to the repository to remove
        """
        def _cleanup():
            path = Path(repo_path)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _cleanup)

    async def cleanup_all(self):
        """Remove all cloned repositories."""
        def _cleanup():
            if self.base_path.exists():
                for item in self.base_path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _cleanup)
