"""
GitHub service for creating branches, commits, and pull requests.
Uses PyGithub to interact with the GitHub API.
"""

import base64
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from github import Github, GithubException
from github.Repository import Repository


@dataclass
class PRResult:
    """Result of a Pull Request creation."""
    success: bool
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    branch_name: Optional[str] = None
    error: Optional[str] = None


class GitHubService:
    """Service for GitHub operations: creating branches, commits, and PRs."""

    def __init__(self, token: str):
        """
        Initialize GitHub service with OAuth token.
        
        Args:
            token: GitHub OAuth token with 'repo' scope
        """
        self.github = Github(token)
        self.token = token

    def _parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        """
        Extract owner and repo name from GitHub URL.
        
        Args:
            repo_url: Full GitHub repository URL
            
        Returns:
            Tuple of (owner, repo_name)
        """
        # Handle various URL formats
        url = repo_url.replace(".git", "")
        
        if "github.com" in url:
            # https://github.com/owner/repo or git@github.com:owner/repo
            if ":" in url and "@" in url:
                # SSH format
                parts = url.split(":")[-1].split("/")
            else:
                # HTTPS format
                parts = url.split("github.com/")[-1].split("/")
            
            if len(parts) >= 2:
                return parts[0], parts[1]
        
        raise ValueError(f"Invalid GitHub URL: {repo_url}")

    def _get_repo(self, repo_url: str) -> Repository:
        """Get PyGithub Repository object from URL."""
        owner, repo_name = self._parse_repo_url(repo_url)
        return self.github.get_repo(f"{owner}/{repo_name}")

    def _generate_branch_name(self) -> str:
        """Generate unique branch name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"codein/docs-{timestamp}"

    async def create_documentation_pr(
        self,
        repo_url: str,
        documentation_files: dict[str, str],
        base_branch: str = "main",
        pr_title: str = "📚 Add AI-generated documentation",
        pr_body: Optional[str] = None,
    ) -> PRResult:
        """
        Create a Pull Request with documentation files.
        
        Args:
            repo_url: GitHub repository URL
            documentation_files: Dict mapping file paths to content
            base_branch: Base branch to create PR against (default: main)
            pr_title: Title for the Pull Request
            pr_body: Body/description for the Pull Request
            
        Returns:
            PRResult with success status and PR details
        """
        try:
            repo = self._get_repo(repo_url)
            
            # Try to get default branch if base_branch doesn't exist
            try:
                base_ref = repo.get_branch(base_branch)
            except GithubException:
                # Fallback to repo's default branch
                base_branch = repo.default_branch
                base_ref = repo.get_branch(base_branch)
            
            # Create new branch
            branch_name = self._generate_branch_name()
            base_sha = base_ref.commit.sha
            
            # Create the branch reference
            ref_name = f"refs/heads/{branch_name}"
            repo.create_git_ref(ref=ref_name, sha=base_sha)
            print(f"✅ Created branch: {branch_name}")
            
            # Commit all documentation files
            for file_path, content in documentation_files.items():
                try:
                    # Check if file already exists
                    try:
                        existing_file = repo.get_contents(file_path, ref=branch_name)
                        # Update existing file
                        repo.update_file(
                            path=file_path,
                            message=f"docs: update {file_path}",
                            content=content,
                            sha=existing_file.sha,
                            branch=branch_name,
                        )
                        print(f"📝 Updated: {file_path}")
                    except GithubException:
                        # Create new file
                        repo.create_file(
                            path=file_path,
                            message=f"docs: add {file_path}",
                            content=content,
                            branch=branch_name,
                        )
                        print(f"📄 Created: {file_path}")
                except GithubException as e:
                    print(f"⚠️ Failed to create {file_path}: {e}")
                    continue
            
            # Generate PR body if not provided
            if not pr_body:
                file_list = "\n".join([f"- `{path}`" for path in documentation_files.keys()])
                pr_body = f"""## 🤖 Auto-generated Documentation

This Pull Request was automatically created by [Code-In](https://github.com/your-org/code-in) AI agent.

### 📁 Files Added/Updated

{file_list}

### 📋 What's Included

- **Architecture Overview**: High-level system design and components
- **Getting Started Guide**: Setup and installation instructions  
- **API Documentation**: Endpoints, methods, and usage examples
- **Code Patterns**: Detected patterns and best practices
- **Dependency Graph**: Visual representation of module dependencies

---

*Generated with ❤️ by Code-In AI*
"""

            # Create Pull Request
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base=base_branch,
            )
            
            print(f"✅ Created PR #{pr.number}: {pr.html_url}")
            
            return PRResult(
                success=True,
                pr_url=pr.html_url,
                pr_number=pr.number,
                branch_name=branch_name,
            )

        except GithubException as e:
            error_msg = f"GitHub API error: {e.data.get('message', str(e))}"
            print(f"❌ {error_msg}")
            return PRResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            return PRResult(success=False, error=error_msg)

    async def check_repo_access(self, repo_url: str) -> tuple[bool, str]:
        """
        Check if we have write access to the repository.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Tuple of (has_access, message)
        """
        try:
            repo = self._get_repo(repo_url)
            
            # Check permissions
            permissions = repo.permissions
            if permissions.push:
                return True, "Write access confirmed"
            else:
                return False, "No write access to this repository"
                
        except GithubException as e:
            return False, f"Cannot access repository: {e.data.get('message', str(e))}"
        except Exception as e:
            return False, f"Error checking access: {str(e)}"

    def get_user_info(self) -> dict:
        """Get authenticated user information."""
        try:
            user = self.github.get_user()
            return {
                "login": user.login,
                "name": user.name,
                "email": user.email,
                "avatar_url": user.avatar_url,
            }
        except GithubException as e:
            return {"error": str(e)}
