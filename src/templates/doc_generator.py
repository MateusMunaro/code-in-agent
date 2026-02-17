"""
Multi-Layer Documentation Generator.

Orchestrates the generation of a stratified documentation system
designed for maximum AI agent performance:

Root Level   → llms.txt, AGENTS.md, repomap.txt
Module Level → {module}/ReadMe.LLM, {module}/AGENTS.md
Semantic     → _codein/ (SCIP, TreeFrag, Knowledge Graph) — Phase 3

This replaces the previous flat docs/ structure with a multi-layer,
polyglot (human/machine) output.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .root_level import (
    LlmsTxtGenerator,
    AgentsMdGenerator,
    RepomapGenerator,
    ProjectProfile,
)
from .module_level import ReadmeLlmGenerator, NestedAgentsMdGenerator
from .mermaid_generator import MermaidGenerator, MermaidContext


# ═══════════════════════════════════════════════════════════════
#  AnalysisResult (unchanged contract with the agent graph)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    """Results from codebase analysis — bridge between agent graph and doc generators."""
    project_name: str
    architecture_pattern: str
    confidence: float
    main_language: str
    framework: Optional[str]
    patterns_detected: list
    files_read: list
    tech_stack: dict
    directory_structure: str
    dependency_graph: dict
    improvements: list
    entry_points: list
    key_modules: list
    # Rich AST data for data-driven documentation
    file_tree: list = None
    code_chunks: list = None
    config_files_content: dict = None


# ═══════════════════════════════════════════════════════════════
#  MultiLayerDocGenerator
# ═══════════════════════════════════════════════════════════════

class MultiLayerDocGenerator:
    """
    Generates the complete multi-layer documentation output.

    Output hierarchy:
        llms.txt                   — navigational index
        AGENTS.md                  — behavioral contract (root)
        repomap.txt                — AST-compressed repo map
        {module}/ReadMe.LLM        — per-module signatures + I/O
        {module}/AGENTS.md         — per-module rule overrides
        _codein/*                  — semantic layer (Phase 3)
    """

    # Directories to skip when discovering modules
    SKIP_DIRS = frozenset({
        "node_modules", "__pycache__", ".git", "venv", "dist",
        "build", ".next", ".cache", ".tox", "coverage",
        "htmlcov", ".mypy_cache", ".pytest_cache", "egg-info",
    })

    # Minimum files in a directory to qualify as a module
    MIN_MODULE_FILES = 1

    def __init__(self, analysis: AnalysisResult):
        self.analysis = analysis
        self.date = datetime.now().strftime("%B %d, %Y")
        self.profile = self._detect_project_profile()
        self._module_map: dict[str, list[dict]] = {}
        self._test_map: dict[str, list[dict]] = {}
        self._build_module_maps()

        # Log initialization
        print(f"[MultiLayerDocGen] 📊 Initialized for: {analysis.project_name}")
        print(f"[MultiLayerDocGen]    file_tree: {len(analysis.file_tree or [])} files")
        print(f"[MultiLayerDocGen]    modules discovered: {len(self._module_map)}")
        print(f"[MultiLayerDocGen]    dep_graph nodes: {len(analysis.dependency_graph.get('nodes', []))}")
        print(f"[MultiLayerDocGen]    dep_graph edges: {len(analysis.dependency_graph.get('edges', []))}")
        print(f"[MultiLayerDocGen]    profile: {self.profile}")

    # =========================================
    # PUBLIC API
    # =========================================

    def generate_full_documentation(self) -> dict[str, str]:
        """
        Generate all documentation files for the new multi-layer architecture.

        Returns:
            Dictionary mapping file paths to content.
            Keys use repository-relative paths (no leading slash).
        """
        docs: dict[str, str] = {}

        # ---- Root Level ----
        docs.update(self._generate_root_level())

        # ---- Module Level ----
        docs.update(self._generate_module_level())

        # ---- Semantic Layer (Phase 3 placeholder) ----
        # Will be populated by scip_service, treefrag_service, knowledge_graph_service

        print(f"[MultiLayerDocGen] ✅ Generated {len(docs)} documentation files")
        return docs

    def generate_summary_documentation(self) -> str:
        """
        Generate a single combined documentation string.
        Used for backward-compatible storage in analysis_results.
        """
        docs = self.generate_full_documentation()
        return self._create_summary_from_docs(docs)

    # =========================================
    # ROOT LEVEL GENERATION
    # =========================================

    def _generate_root_level(self) -> dict[str, str]:
        """Generate llms.txt, AGENTS.md, repomap.txt."""
        docs: dict[str, str] = {}
        module_paths = sorted(self._module_map.keys())

        # llms.txt
        llms_gen = LlmsTxtGenerator(
            project_name=self.analysis.project_name,
            architecture_pattern=self.analysis.architecture_pattern,
            main_language=self.analysis.main_language,
            framework=self.analysis.framework,
            tech_stack=self.analysis.tech_stack,
            entry_points=self.analysis.entry_points,
            key_modules=self.analysis.key_modules,
            module_paths=module_paths,
            profile=self.profile,
        )
        docs["llms.txt"] = llms_gen.generate()

        # AGENTS.md (root)
        agents_gen = AgentsMdGenerator(
            project_name=self.analysis.project_name,
            architecture_pattern=self.analysis.architecture_pattern,
            main_language=self.analysis.main_language,
            framework=self.analysis.framework,
            patterns_detected=self.analysis.patterns_detected,
            tech_stack=self.analysis.tech_stack,
            entry_points=self.analysis.entry_points,
            key_modules=self.analysis.key_modules,
            profile=self.profile,
            module_paths=module_paths,
        )
        docs["AGENTS.md"] = agents_gen.generate()

        # repomap.txt
        repomap_gen = RepomapGenerator(
            project_name=self.analysis.project_name,
            file_tree=self.analysis.file_tree,
            dependency_graph=self.analysis.dependency_graph,
            main_language=self.analysis.main_language,
            profile=self.profile,
        )
        docs["repomap.txt"] = repomap_gen.generate()

        return docs

    # =========================================
    # MODULE LEVEL GENERATION
    # =========================================

    def _generate_module_level(self) -> dict[str, str]:
        """Generate ReadMe.LLM and AGENTS.md for each discovered module."""
        docs: dict[str, str] = {}

        for mod_path, mod_files in sorted(self._module_map.items()):
            # Find test files related to this module
            test_files = self._find_tests_for_module(mod_path)

            # ReadMe.LLM
            readme_gen = ReadmeLlmGenerator(
                module_path=mod_path,
                module_files=mod_files,
                test_files=test_files,
                dependency_graph=self.analysis.dependency_graph,
                main_language=self.analysis.main_language,
                project_name=self.analysis.project_name,
            )
            docs[f"{mod_path}/ReadMe.LLM"] = readme_gen.generate()

            # AGENTS.md (nested)
            agents_gen = NestedAgentsMdGenerator(
                module_path=mod_path,
                module_files=mod_files,
                main_language=self.analysis.main_language,
                project_name=self.analysis.project_name,
                architecture_pattern=self.analysis.architecture_pattern,
            )
            docs[f"{mod_path}/AGENTS.md"] = agents_gen.generate()

        return docs

    # =========================================
    # MODULE DISCOVERY
    # =========================================

    def _build_module_maps(self):
        """
        Group file_tree entries by parent directory to discover modules.
        Also builds a parallel map of test files.
        """
        file_tree = self.analysis.file_tree or []

        for f in file_tree:
            path = f.get("path", "")
            if not path:
                continue

            parts = Path(path).parts

            # Skip noise directories
            if any(skip in parts for skip in self.SKIP_DIRS):
                continue

            # Determine parent directory (1 or 2 levels)
            if len(parts) <= 1:
                # Root-level file — goes into a virtual "root" module only
                # if we have enough root files
                continue

            # Use the first 1-2 directory levels as module path
            if len(parts) == 2:
                mod_path = parts[0]
            else:
                # Use first two levels for deeper structures
                mod_path = str(Path(parts[0]) / parts[1])

            # Classify as test or source
            path_lower = path.lower()
            is_test = any(
                kw in path_lower
                for kw in ("test", "spec", "__tests__", "tests/")
            )

            if is_test:
                self._test_map.setdefault(mod_path, []).append(f)
            else:
                self._module_map.setdefault(mod_path, []).append(f)

        # Filter out modules with too few files
        self._module_map = {
            k: v for k, v in self._module_map.items()
            if len(v) >= self.MIN_MODULE_FILES
        }

    def _find_tests_for_module(self, module_path: str) -> list[dict]:
        """
        Find test files that correspond to a given module.

        Matches by:
        1. Direct: tests in the same module path
        2. Sibling: tests/ directory at the same level
        3. Name: test files whose names reference the module
        """
        # Direct match
        if module_path in self._test_map:
            return self._test_map[module_path]

        # Try common test directory patterns
        parts = Path(module_path).parts
        test_candidates = [
            f"tests/{parts[-1]}" if len(parts) >= 1 else "",
            f"{parts[0]}/tests" if len(parts) >= 1 else "",
            f"test/{parts[-1]}" if len(parts) >= 1 else "",
        ]

        for candidate in test_candidates:
            if candidate and candidate in self._test_map:
                return self._test_map[candidate]

        return []

    # =========================================
    # PROJECT PROFILE DETECTION
    # =========================================

    def _detect_project_profile(self) -> ProjectProfile:
        """Detect project characteristics to adapt documentation."""
        lang = (self.analysis.main_language or "").lower()
        file_tree = self.analysis.file_tree or []
        config = self.analysis.config_files_content or {}

        all_funcs = self._get_all_functions()
        all_classes = self._get_all_classes()

        profile = ProjectProfile(
            has_classes=len(all_classes) > 0,
            naming_convention=self._detect_naming_convention(all_funcs),
            total_functions=len(all_funcs),
            total_classes=len(all_classes),
            total_files=len(file_tree),
        )

        # Detect build system from config files
        config_names = [Path(k).name.lower() for k in config.keys()]

        if any("makefile" in n for n in config_names):
            profile.has_build_system = True
            profile.build_tool = "Make"
            for k, v in config.items():
                if "makefile" in k.lower():
                    profile.build_commands = self._extract_makefile_targets(v)
                    break
        elif any("cmakelists" in n for n in config_names):
            profile.has_build_system = True
            profile.build_tool = "CMake"
        elif any("package.json" in n for n in config_names):
            profile.has_build_system = True
            profile.build_tool = "npm"
            for k, v in config.items():
                if "package.json" in k.lower():
                    profile.build_commands = self._extract_npm_scripts(v)
                    break
        elif any("cargo.toml" in n for n in config_names):
            profile.has_build_system = True
            profile.build_tool = "Cargo"
            profile.build_commands = {
                "build": "cargo build",
                "run": "cargo run",
                "test": "cargo test",
            }
        elif any("go.mod" in n for n in config_names):
            profile.has_build_system = True
            profile.build_tool = "Go"
            profile.build_commands = {
                "build": "go build",
                "run": "go run",
                "test": "go test ./...",
            }
        elif any(
            "pyproject.toml" in n or "requirements.txt" in n
            for n in config_names
        ):
            profile.has_build_system = True
            profile.build_tool = "pip/poetry"

        # Detect test presence
        for f in file_tree:
            p = f.get("path", "").lower()
            if "test" in p or "spec" in p or "_test." in p or "test_" in p:
                profile.has_tests = True
                break

        # Detect project type
        all_paths = " ".join(f.get("path", "").lower() for f in file_tree)
        if any(x in all_paths for x in ("routes/", "views/", "handlers/", "api/", "server.", "app.")):
            profile.is_web = True
        if any(x in all_paths for x in ("cli.", "cmd/", "main.", "__main__.")):
            profile.is_cli = True
        if any(x in all_paths for x in ("lib/", "include/", "pkg/")):
            profile.is_library = True

        # Doc style
        has_docstrings = any(f.get("docstring") for f in all_funcs)
        if has_docstrings:
            profile.doc_style = "docstrings"

        # Structs
        if lang in ("c", "c++", "rust", "go"):
            profile.has_structs = True

        return profile

    # =========================================
    # UTILITY METHODS
    # =========================================

    def _get_all_functions(self) -> list:
        """Get all function details from file_tree."""
        functions = []
        for f in (self.analysis.file_tree or []):
            for func in f.get("function_details", []):
                if isinstance(func, dict):
                    func["_file_path"] = f.get("path", "")
                    functions.append(func)
        return functions

    def _get_all_classes(self) -> list:
        """Get all class details from file_tree."""
        classes = []
        for f in (self.analysis.file_tree or []):
            for cls in f.get("class_details", []):
                if isinstance(cls, dict):
                    cls["_file_path"] = f.get("path", "")
                    classes.append(cls)
        return classes

    def _detect_naming_convention(self, functions: list) -> str:
        """Detect the dominant naming convention from function names."""
        import re
        snake = 0
        camel = 0
        pascal = 0

        for func in functions[:50]:
            name = func.get("name", "")
            if not name or name.startswith("_"):
                name = name.lstrip("_")
            if not name:
                continue
            if "_" in name:
                snake += 1
            elif name[0].isupper():
                pascal += 1
            elif any(c.isupper() for c in name[1:]):
                camel += 1
            else:
                snake += 1

        if pascal > camel and pascal > snake:
            return "PascalCase"
        elif camel > snake:
            return "camelCase"
        return "snake_case"

    def _extract_makefile_targets(self, content: str) -> dict:
        """Extract targets from a Makefile."""
        import re
        targets = {}
        reserved = {"else", "endif", "ifdef", "ifndef", "ifeq", "ifneq"}
        for match in re.finditer(r'^([a-zA-Z_][a-zA-Z0-9_-]*)\s*:', content, re.MULTILINE):
            target = match.group(1)
            if target not in reserved:
                targets[target] = f"make {target}"
        return targets

    def _extract_npm_scripts(self, content: str) -> dict:
        """Extract scripts from package.json content."""
        try:
            pkg = json.loads(content)
            scripts = pkg.get("scripts", {})
            return {name: f"npm run {name}" for name in scripts.keys()}
        except Exception:
            return {}

    def _create_summary_from_docs(self, docs: dict[str, str]) -> str:
        """
        Create a single summary string from the multi-layer docs.
        Used for backward-compatible storage.
        """
        priority = ["AGENTS.md", "llms.txt", "repomap.txt"]

        parts: list[str] = []

        # Priority files first
        for key in priority:
            if key in docs:
                parts.append(docs[key])

        # Then module ReadMe.LLM files
        module_docs = sorted(k for k in docs if k.endswith("/ReadMe.LLM"))
        for key in module_docs[:10]:
            parts.append(docs[key])

        # Mention remaining files
        remaining = [
            k for k in docs
            if k not in priority and k not in module_docs[:10]
        ]
        if remaining:
            parts.append(
                f"\n---\n\n## Additional Files\n\n"
                f"*{len(remaining)} additional documentation files available in storage.*\n\n"
                f"### Available Files:\n"
                + "\n".join(f"- `{f}`" for f in sorted(remaining))
            )

        return "\n\n---\n\n".join(parts) if parts else "No documentation generated."


# ═══════════════════════════════════════════════════════════════
#  Backward-compatible aliases
# ═══════════════════════════════════════════════════════════════

# Keep the old class name available for imports that haven't updated yet
DocumentationGenerator = MultiLayerDocGenerator


def generate_documentation(
    project_name: str,
    architecture_pattern: str,
    confidence: float,
    main_language: str,
    files_read: list,
    patterns_detected: list,
    dependency_graph: dict,
    directory_structure: str = "",
    framework: str = None,
    tech_stack: dict = None,
    improvements: list = None,
    entry_points: list = None,
    key_modules: list = None,
    file_tree: list = None,
    code_chunks: list = None,
    config_files_content: dict = None,
    output_format: str = "summary",
) -> str | dict[str, str]:
    """
    Convenience function to generate documentation.

    Args:
        project_name: Name of the project
        architecture_pattern: Detected architecture
        confidence: Analysis confidence (0-1)
        main_language: Main programming language
        files_read: List of files analyzed
        patterns_detected: List of detected patterns
        dependency_graph: Dependency graph data
        directory_structure: Directory tree string
        framework: Framework used
        tech_stack: Technology stack
        improvements: List of improvement suggestions
        entry_points: List of entry point files
        key_modules: List of key modules
        file_tree: List of FileInfo dicts with function_details, class_details
        code_chunks: List of CodeChunk data from embeddings
        config_files_content: Dict of config file paths to their content
        output_format: "summary" for single file, "full" for complete structure

    Returns:
        Single markdown string or dict of file paths to content
    """
    analysis = AnalysisResult(
        project_name=project_name,
        architecture_pattern=architecture_pattern,
        confidence=confidence,
        main_language=main_language,
        framework=framework,
        patterns_detected=patterns_detected or [],
        files_read=files_read or [],
        tech_stack=tech_stack or {},
        directory_structure=directory_structure or "",
        dependency_graph=dependency_graph or {},
        improvements=improvements or [],
        entry_points=entry_points or [],
        key_modules=key_modules or [],
        file_tree=file_tree or [],
        code_chunks=code_chunks or [],
        config_files_content=config_files_content or {},
    )

    generator = MultiLayerDocGenerator(analysis)

    if output_format == "full":
        return generator.generate_full_documentation()
    else:
        return generator.generate_summary_documentation()
