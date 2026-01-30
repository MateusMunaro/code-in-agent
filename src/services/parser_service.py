"""
Parser service using tree-sitter for AST analysis.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

# Tree-sitter imports
try:
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class FileInfo:
    """Information about a parsed file."""
    path: str
    language: str
    size_bytes: int
    line_count: int
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert FileInfo to dictionary."""
        return asdict(self)


# Language detection by extension
LANGUAGE_MAP = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
}

# Config files that indicate project structure
CONFIG_FILES = {
    "package.json": "nodejs",
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "setup.py": "python",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "pom.xml": "java",
    "build.gradle": "java",
    "Gemfile": "ruby",
    "composer.json": "php",
    ".csproj": "csharp",
    "mix.exs": "elixir",
    "deno.json": "deno",
    "bun.lockb": "bun",
}


class ParserService:
    """Service for parsing code files and extracting structure."""

    def __init__(self):
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._initialize_parsers()

    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        # Note: Full tree-sitter setup requires language-specific parsers
        # This is a simplified version for demonstration
        pass

    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect the programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return LANGUAGE_MAP.get(ext)

    async def parse_repository(self, repo_path: str) -> list[dict]:
        """
        Parse all code files in a repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of FileInfo dicts for each parsed file
        """
        files_info = []
        repo_root = Path(repo_path)

        # Exclusion patterns
        exclude_dirs = {
            "node_modules", "__pycache__", ".git", ".venv", "venv",
            "dist", "build", ".next", "target", ".idea", ".vscode",
            "vendor", "coverage", ".pytest_cache", ".mypy_cache",
            ".tox", "eggs", ".eggs", "*.egg-info",
        }

        def _parse_files():
            for file_path in repo_root.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip excluded directories
                rel_parts = file_path.relative_to(repo_root).parts
                if any(exc in rel_parts for exc in exclude_dirs):
                    continue

                # Check if it's a code file
                language = self._detect_language(str(file_path))
                if not language:
                    continue

                try:
                    # Get file stats
                    stat = file_path.stat()
                    
                    # Skip large files (> 500KB)
                    if stat.st_size > 500 * 1024:
                        continue

                    # Read and analyze file
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    rel_path = str(file_path.relative_to(repo_root))
                    line_count = content.count("\n") + 1

                    # Extract basic structure
                    file_info = self._analyze_file(rel_path, language, content, stat.st_size, line_count)
                    files_info.append(file_info)

                except Exception as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")
                    continue

            return files_info

        loop = asyncio.get_event_loop()
        files_info_objects = await loop.run_in_executor(None, _parse_files)
        # Convert FileInfo objects to dicts for JSON serialization and subscript access
        return [f.to_dict() for f in files_info_objects]

    def _analyze_file(
        self,
        file_path: str,
        language: str,
        content: str,
        size_bytes: int,
        line_count: int
    ) -> FileInfo:
        """Analyze a single file and extract structure."""
        
        file_info = FileInfo(
            path=file_path,
            language=language,
            size_bytes=size_bytes,
            line_count=line_count,
        )

        # Language-specific parsing
        if language == "python":
            self._parse_python(content, file_info)
        elif language in ("javascript", "typescript"):
            self._parse_javascript(content, file_info)
        elif language == "java":
            self._parse_java(content, file_info)
        elif language == "go":
            self._parse_go(content, file_info)

        return file_info

    def _parse_python(self, content: str, file_info: FileInfo):
        """Extract Python imports, functions, and classes."""
        import re

        # Find imports
        import_patterns = [
            r"^import\s+(\S+)",
            r"^from\s+(\S+)\s+import",
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                file_info.imports.append(match.group(1))

        # Find function definitions
        func_pattern = r"^def\s+(\w+)\s*\("
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            file_info.functions.append(match.group(1))

        # Find class definitions
        class_pattern = r"^class\s+(\w+)"
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            file_info.classes.append(match.group(1))

    def _parse_javascript(self, content: str, file_info: FileInfo):
        """Extract JavaScript/TypeScript imports, functions, and classes."""
        import re

        # Find imports
        import_patterns = [
            r"import\s+.*?\s+from\s+['\"](.+?)['\"]",
            r"require\s*\(\s*['\"](.+?)['\"]\s*\)",
            r"import\s*\(\s*['\"](.+?)['\"]\s*\)",
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                file_info.imports.append(match.group(1))

        # Find exports
        export_patterns = [
            r"export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)",
            r"export\s+\{\s*([^}]+)\s*\}",
        ]
        for pattern in export_patterns:
            for match in re.finditer(pattern, content):
                exports = match.group(1).split(",")
                file_info.exports.extend([e.strip().split(" ")[0] for e in exports])

        # Find function definitions
        func_patterns = [
            r"function\s+(\w+)\s*\(",
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?function",
        ]
        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                file_info.functions.append(match.group(1))

        # Find class definitions
        class_pattern = r"class\s+(\w+)"
        for match in re.finditer(class_pattern, content):
            file_info.classes.append(match.group(1))

    def _parse_java(self, content: str, file_info: FileInfo):
        """Extract Java imports and classes."""
        import re

        # Find imports
        import_pattern = r"^import\s+(?:static\s+)?(.+?);"
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            file_info.imports.append(match.group(1))

        # Find class definitions
        class_pattern = r"(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)"
        for match in re.finditer(class_pattern, content):
            file_info.classes.append(match.group(1))

        # Find method definitions
        method_pattern = r"(?:public|private|protected)\s+(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\("
        for match in re.finditer(method_pattern, content):
            file_info.functions.append(match.group(1))

    def _parse_go(self, content: str, file_info: FileInfo):
        """Extract Go imports and functions."""
        import re

        # Find imports
        import_patterns = [
            r'import\s+"(.+?)"',
            r'import\s+\w+\s+"(.+?)"',
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                file_info.imports.append(match.group(1))

        # Find imports block
        import_block = r"import\s*\(([\s\S]*?)\)"
        block_match = re.search(import_block, content)
        if block_match:
            block_content = block_match.group(1)
            for line in block_content.split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    # Extract package path from quoted string
                    match = re.search(r'"(.+?)"', line)
                    if match:
                        file_info.imports.append(match.group(1))

        # Find function definitions
        func_pattern = r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\("
        for match in re.finditer(func_pattern, content):
            file_info.functions.append(match.group(1))

    async def detect_project_type(self, repo_path: str) -> dict:
        """
        Detect the project type based on config files.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary with project type information
        """
        repo_root = Path(repo_path)
        detected = {
            "primary_language": None,
            "frameworks": [],
            "config_files": [],
            "has_tests": False,
            "has_docs": False,
        }

        def _detect():
            # Check for config files
            for config_file, lang in CONFIG_FILES.items():
                if (repo_root / config_file).exists():
                    detected["config_files"].append(config_file)
                    if not detected["primary_language"]:
                        detected["primary_language"] = lang

            # Check for test directories
            test_dirs = ["test", "tests", "__tests__", "spec", "specs"]
            for test_dir in test_dirs:
                if (repo_root / test_dir).exists():
                    detected["has_tests"] = True
                    break

            # Check for docs
            doc_dirs = ["docs", "doc", "documentation"]
            for doc_dir in doc_dirs:
                if (repo_root / doc_dir).exists():
                    detected["has_docs"] = True
                    break

            # Detect frameworks from package.json
            package_json = repo_root / "package.json"
            if package_json.exists():
                import json
                try:
                    with open(package_json, "r") as f:
                        pkg = json.load(f)
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                    
                    framework_indicators = {
                        "react": "React",
                        "next": "Next.js",
                        "vue": "Vue.js",
                        "nuxt": "Nuxt",
                        "@angular/core": "Angular",
                        "svelte": "Svelte",
                        "express": "Express",
                        "fastify": "Fastify",
                        "elysia": "Elysia",
                        "hono": "Hono",
                        "nestjs": "NestJS",
                    }
                    
                    for dep, framework in framework_indicators.items():
                        if dep in deps:
                            detected["frameworks"].append(framework)
                except Exception:
                    pass

            return detected

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _detect)
