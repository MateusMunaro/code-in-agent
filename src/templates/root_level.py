"""
Root Level Output Generators.

Generates the top-level governance and navigation artifacts:
- /llms.txt       — LLM-navigable index of all documentation
- /AGENTS.md      — Behavioral contract (Rules, Commands, Boundaries)
- /repomap.txt    — AST-compressed repository map (high-level symbols only)

These files sit at the repository root and serve as the primary entry
points for AI coding agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ProjectProfile:
    """Detected project characteristics used across all generators."""
    has_classes: bool = False
    has_structs: bool = False
    has_build_system: bool = False
    build_tool: Optional[str] = None
    build_commands: dict = field(default_factory=dict)
    has_tests: bool = False
    is_library: bool = False
    is_cli: bool = False
    is_web: bool = False
    naming_convention: str = "snake_case"
    doc_style: Optional[str] = None
    total_functions: int = 0
    total_classes: int = 0
    total_files: int = 0


# ═══════════════════════════════════════════════════════════════
#  llms.txt  —  Navigational Index
# ═══════════════════════════════════════════════════════════════

class LlmsTxtGenerator:
    """
    Generates ``llms.txt`` following the llms-txt spec.

    The file acts as a **table of contents** that tells an LLM
    which artifacts exist and how to reach them.  It is intentionally
    small (fits in a single context window) and links out to heavier
    documents for deep dives.
    """

    def __init__(
        self,
        project_name: str,
        architecture_pattern: str,
        main_language: str,
        framework: Optional[str],
        tech_stack: dict,
        entry_points: list,
        key_modules: list,
        module_paths: list[str],
        profile: ProjectProfile,
    ):
        self.project_name = project_name
        self.architecture = architecture_pattern
        self.language = main_language
        self.framework = framework
        self.tech_stack = tech_stack
        self.entry_points = entry_points
        self.key_modules = key_modules
        self.module_paths = module_paths
        self.profile = profile
        self.date = datetime.now().strftime("%Y-%m-%d")

    def generate(self) -> str:
        """Return the full content of ``llms.txt``."""
        sections = [
            self._header(),
            self._project_summary(),
            self._documentation_map(),
            self._module_index(),
            self._quick_reference(),
        ]
        return "\n".join(sections)

    # ---- sections ----

    def _header(self) -> str:
        return f"""# {self.project_name}

> llms.txt — Machine-readable documentation index
> Generated: {self.date}
"""

    def _project_summary(self) -> str:
        fw = f" + {self.framework}" if self.framework else ""
        stack_items = ", ".join(
            f"{k}: {v}" for k, v in list(self.tech_stack.items())[:6]
        ) if self.tech_stack else "See repomap.txt"

        project_type_parts = []
        if self.profile.is_web:
            project_type_parts.append("web")
        if self.profile.is_cli:
            project_type_parts.append("cli")
        if self.profile.is_library:
            project_type_parts.append("library")
        project_type = ", ".join(project_type_parts) if project_type_parts else "application"

        return f"""## Project Summary

- Architecture: {self.architecture}
- Language: {self.language}{fw}
- Type: {project_type}
- Files: {self.profile.total_files} | Functions: {self.profile.total_functions} | Classes: {self.profile.total_classes}
- Stack: {stack_items}
"""

    def _documentation_map(self) -> str:
        lines = [
            "## Documentation Map",
            "",
            "| Artifact | Path | Purpose |",
            "|----------|------|---------|",
            "| Behavioral Contract | [AGENTS.md](AGENTS.md) | Rules, commands, boundaries for AI agents |",
            "| Repository Map | [repomap.txt](repomap.txt) | AST-compressed symbol map of entire repo |",
        ]

        # Module-level docs
        for mod in self.module_paths[:20]:
            lines.append(
                f"| Module Doc | [{mod}/ReadMe.LLM]({mod}/ReadMe.LLM) "
                f"| Signatures + I/O for `{mod}/` |"
            )
            lines.append(
                f"| Module Rules | [{mod}/AGENTS.md]({mod}/AGENTS.md) "
                f"| Module-specific overrides |"
            )

        # Semantic layer
        lines.extend([
            "| SCIP Index | [_codein/scip_index.scip](_codein/scip_index.scip) | Compiler-precision symbol navigation |",
            "| TreeFrag | [_codein/treefrag.txt](_codein/treefrag.txt) | Hierarchical LOD representation |",
            "| Knowledge Graph | [_codein/knowledge_graph.json](_codein/knowledge_graph.json) | Enriched dependency graph |",
        ])

        lines.append("")
        return "\n".join(lines)

    def _module_index(self) -> str:
        if not self.key_modules:
            return ""

        lines = ["## Key Modules", ""]
        for mod in self.key_modules[:15]:
            name = mod if isinstance(mod, str) else mod.get("name", "Unknown")
            lines.append(f"- `{name}/`")
        lines.append("")
        return "\n".join(lines)

    def _quick_reference(self) -> str:
        lines = ["## Quick Reference", ""]

        if self.entry_points:
            lines.append("### Entry Points")
            for ep in self.entry_points[:8]:
                lines.append(f"- `{ep}`")
            lines.append("")

        if self.profile.build_commands:
            lines.append("### Build Commands")
            lines.append("```")
            for name, cmd in list(self.profile.build_commands.items())[:8]:
                lines.append(f"{cmd}  # {name}")
            lines.append("```")
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  AGENTS.md  —  Behavioral Contract
# ═══════════════════════════════════════════════════════════════

class AgentsMdGenerator:
    """
    Generates the root ``AGENTS.md`` — the behavioral contract for
    AI coding agents working in this repository.

    Covers:
    - Navigation protocol (what to read and in which order)
    - File structure rules (architecture-specific)
    - Code pattern rules
    - Naming conventions (language-specific)
    - Import ordering
    - Anti-patterns / prohibitions
    - Pre-commit checklist
    """

    def __init__(
        self,
        project_name: str,
        architecture_pattern: str,
        main_language: str,
        framework: Optional[str],
        patterns_detected: list,
        tech_stack: dict,
        entry_points: list,
        key_modules: list,
        profile: ProjectProfile,
        module_paths: list[str],
    ):
        self.project_name = project_name
        self.architecture = architecture_pattern
        self.language = main_language
        self.framework = framework
        self.patterns = patterns_detected or []
        self.tech_stack = tech_stack
        self.entry_points = entry_points or []
        self.key_modules = key_modules or []
        self.profile = profile
        self.module_paths = module_paths or []
        self.date = datetime.now().strftime("%B %Y")

    def generate(self) -> str:
        sections = [
            self._header(),
            self._navigation_protocol(),
            self._project_context(),
            self._file_structure_rules(),
            self._code_patterns(),
            self._naming_conventions(),
            self._import_rules(),
            self._commands(),
            self._anti_patterns(),
            self._checklist(),
            self._key_modules_section(),
            self._entry_points_section(),
            self._footer(),
        ]
        return "\n".join(s for s in sections if s)

    # ---- header ----

    def _header(self) -> str:
        return f"""# AGENTS.md — {self.project_name}

> Behavioral contract for AI coding agents.
> Read this file **BEFORE** making any code changes.

| Field | Value |
|-------|-------|
| Architecture | {self.architecture} |
| Language | {self.language} |
| Framework | {self.framework or 'N/A'} |
| Last Updated | {self.date} |

---
"""

    # ---- navigation ----

    def _navigation_protocol(self) -> str:
        module_tree = ""
        if self.module_paths:
            module_tree = "\n".join(
                f"│   ├── {m}/ReadMe.LLM" for m in self.module_paths[:6]
            )
            module_tree = f"\n{module_tree}"

        return f"""## Navigation Protocol

```
Reading order:
1. THIS FILE (AGENTS.md)          ← You are here
2. llms.txt                       ← Documentation map
3. repomap.txt                    ← Symbol overview
4. <module>/ReadMe.LLM            ← Only the module you will modify
5. <module>/AGENTS.md             ← Module-specific overrides
```

### Decision Tree

```
What do you need to do?
│
├─► Understand overall architecture?
│   └─► Read: repomap.txt → then this file's "File Structure" section
│
├─► Create new component/module?
│   └─► Read: <nearest module>/ReadMe.LLM for patterns
│
├─► Modify existing code?
│   └─► Read: <target module>/ReadMe.LLM + AGENTS.md
│
├─► Add new feature end-to-end?
│   └─► Read: repomap.txt for affected modules → each ReadMe.LLM
│
└─► Debug or trace data flow?
    └─► Read: _codein/knowledge_graph.json or repomap.txt
```

### Rules
- **DO NOT** load all ReadMe.LLM files at once — load only the module you need
- **DO NOT** skip this file and go straight to code
- **DO NOT** modify code without checking the module's AGENTS.md overrides

---
"""

    # ---- project context ----

    def _project_context(self) -> str:
        patterns_list = "\n".join(
            f"- **{p.get('name', 'Unknown')}**: {p.get('description', 'N/A')}"
            for p in self.patterns[:5]
        ) if self.patterns else "- No specific patterns detected"

        tech_list = "\n".join(
            f"- **{k}**: {v}" for k, v in (self.tech_stack or {}).items()
        ) if self.tech_stack else "- Stack not identified"

        stats = []
        if self.profile.total_files:
            stats.append(f"{self.profile.total_files} files")
        if self.profile.total_functions:
            stats.append(f"{self.profile.total_functions} functions")
        if self.profile.total_classes:
            stats.append(f"{self.profile.total_classes} classes")

        return f"""## Project Context

**Codebase:** {', '.join(stats) if stats else 'N/A'}

### Detected Patterns

{patterns_list}

### Technology Stack

{tech_list}

---
"""

    # ---- file structure ----

    def _file_structure_rules(self) -> str:
        arch = self.architecture.lower()

        if "clean" in arch:
            body = self._rules_clean()
        elif "mvc" in arch:
            body = self._rules_mvc()
        elif "hexagonal" in arch or "ports" in arch:
            body = self._rules_hexagonal()
        elif "microservice" in arch:
            body = self._rules_microservices()
        elif "monolith" in arch:
            body = self._rules_monolith()
        else:
            body = self._rules_generic()

        return f"""## File Structure Rules

{body}

---
"""

    def _rules_clean(self) -> str:
        return """### Clean Architecture

```
┌──────────────────────────────────────────┐
│           Frameworks & Drivers           │
│  (Controllers, Routes, DB, External APIs)│
├──────────────────────────────────────────┤
│           Interface Adapters             │
│   (Presenters, Gateways, Repositories)   │
├──────────────────────────────────────────┤
│           Application Business           │
│        (Use Cases, Services)             │
├──────────────────────────────────────────┤
│           Enterprise Business            │
│        (Entities, Domain Logic)          │
└──────────────────────────────────────────┘
```

**Rules:**
1. Dependencies only point **INWARD** (to inner layers)
2. Outer layers are NOT known by inner layers
3. Use Cases orchestrate; Entities contain business logic
4. Interfaces / Contracts belong in inner layers"""

    def _rules_mvc(self) -> str:
        return """### MVC (Model-View-Controller)

```
Models/      → Data logic and validation
Views/       → Presentation only
Controllers/ → Flow orchestration
```

**Rules:**
1. Controllers must not contain business logic
2. Models must not reference Views
3. Views are passive (display only)"""

    def _rules_hexagonal(self) -> str:
        return """### Hexagonal (Ports & Adapters)

```
adapters/
├── inbound/   → HTTP, CLI, Events (input)
└── outbound/  → Database, APIs (output)

domain/        → Business logic (core)

ports/
├── inbound/   → Interfaces for use cases
└── outbound/  → Interfaces for external services
```

**Rules:**
1. Domain **NEVER** imports code from adapters
2. Adapters implement Ports
3. All external communication goes through Ports"""

    def _rules_microservices(self) -> str:
        return """### Microservices

Each service is independent:

```
services/
├── service-a/  (src/, Dockerfile, README.md)
└── service-b/
```

**Rules:**
1. Services communicate via APIs / Events only
2. Each service has its own database
3. Do not share code between services (extract to shared lib if needed)"""

    def _rules_monolith(self) -> str:
        return """### Modular Monolith

```
src/
├── modules/
│   ├── module-a/
│   └── module-b/
├── shared/         → Shared utilities
└── infrastructure/
```

**Rules:**
1. Modules should be as independent as possible
2. Inter-module communication via public interfaces
3. `shared/` contains only truly generic utilities"""

    def _rules_generic(self) -> str:
        return """### Project Structure

Follow the existing folder organization. Before creating new files:

1. Check where similar files are located
2. Maintain consistency with existing patterns
3. Refer to the relevant module's `ReadMe.LLM` for the component catalog"""

    # ---- code patterns ----

    def _code_patterns(self) -> str:
        if not self.patterns:
            return """## Code Patterns

Follow existing patterns in the code. Before implementing:

1. Look for similar implementations in the codebase
2. Read the target module's `ReadMe.LLM`
3. Maintain consistency with existing style

---
"""

        parts = ["## Code Patterns\n"]
        for p in self.patterns[:5]:
            name = p.get("name", "Unknown")
            desc = p.get("description", "")
            evidence = p.get("evidence", [])[:2]
            parts.append(f"### {name}\n")
            parts.append(f"{desc}\n")
            if evidence:
                parts.append("**Evidence:**")
                for e in evidence:
                    parts.append(f"- `{e}`")
            parts.append("")

        parts.append("---\n")
        return "\n".join(parts)

    # ---- naming ----

    def _naming_conventions(self) -> str:
        lang = self.language.lower()

        if lang in ("python", "py"):
            body = self._naming_python()
        elif lang in ("typescript", "javascript", "ts", "js"):
            body = self._naming_js_ts()
        elif lang in ("java", "kotlin"):
            body = self._naming_java()
        else:
            body = (
                "Follow the conventions of the project's main language.\n"
                "Check existing files for examples."
            )

        return f"""## Naming Conventions

{body}

---
"""

    def _naming_python(self) -> str:
        return """```python
# Files / modules: snake_case
my_module.py

# Classes: PascalCase
class MyService: ...

# Functions / variables: snake_case
def my_function():
    my_variable = 1

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3

# Private: prefix _
def _internal(): ...
```"""

    def _naming_js_ts(self) -> str:
        return """```typescript
// Files: kebab-case or PascalCase for components
my-service.ts / MyComponent.tsx

// Classes & Components: PascalCase
class MyService {}
function MyComponent() {}

// Functions & variables: camelCase
function myFunction() {}
const myVariable = 1;

// Constants: UPPER_SNAKE_CASE
const MAX_RETRIES = 3;

// Interfaces / Types: PascalCase
interface UserData {}
type UserRole = 'admin' | 'user';
```"""

    def _naming_java(self) -> str:
        return """```java
// Files: PascalCase (matches class name)
MyService.java

// Classes: PascalCase
class MyService {}

// Methods / variables: camelCase
public void myMethod() {}

// Constants: UPPER_SNAKE_CASE
public static final int MAX_RETRIES = 3;

// Packages: lowercase
package com.example.myapp;
```"""

    # ---- imports ----

    def _import_rules(self) -> str:
        lang = self.language.lower()

        base = """## Import Rules

### Order
1. **Standard libraries** (built-in)
2. **External dependencies** (third-party)
3. **Internal imports** (from project)

"""

        if lang in ("python", "py"):
            base += """```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
from fastapi import FastAPI

# 3. Internal
from .my_module import my_function
from src.services import MyService
```
"""
        elif lang in ("typescript", "javascript", "ts", "js"):
            base += """```typescript
// 1. Node / Built-in
import path from 'path';

// 2. Third-party
import express from 'express';

// 3. Internal — aliases
import { MyService } from '@/services/my-service';

// 4. Internal — relative
import { helper } from './utils';
```
"""

        base += "\n---\n"
        return base

    # ---- commands ----

    def _commands(self) -> str:
        if self.profile.build_commands:
            lines = ["## Useful Commands\n", "```bash"]
            for name, cmd in list(self.profile.build_commands.items())[:10]:
                lines.append(f"# {name}")
                lines.append(cmd)
                lines.append("")
            lines.append("```\n\n---\n")
            return "\n".join(lines)

        lang = self.language.lower()
        if lang in ("python", "py"):
            return """## Useful Commands

```bash
pip install -r requirements.txt   # Install deps
pytest                             # Run tests
mypy src/                          # Type check
black src/ && isort src/           # Format
```

---
"""
        elif lang in ("typescript", "javascript", "ts", "js"):
            return """## Useful Commands

```bash
npm install      # Install deps
npm run dev      # Dev server
npm run build    # Build
npm run test     # Tests
npm run lint     # Lint
```

---
"""
        return ""

    # ---- anti-patterns ----

    def _anti_patterns(self) -> str:
        return """## Anti-Patterns

> Things you must **NOT** do

1. **Don't duplicate code** — check the module's `ReadMe.LLM` for existing implementations
2. **Don't ignore architecture** — respect layers and boundaries described above
3. **Don't hardcode values** — use configuration / constants / env vars
4. **Don't make giant commits** — small, focused, one feature per commit
5. **Don't ignore types** — maintain the project's type discipline
6. **Don't modify build config** unless strictly necessary

---
"""

    # ---- checklist ----

    def _checklist(self) -> str:
        return """## Pre-Commit Checklist

```
[ ] Code follows patterns from the module's ReadMe.LLM
[ ] New files are in the correct directory (check File Structure Rules)
[ ] Imports follow the ordering convention
[ ] No duplicate code of something that already exists
[ ] Tests added / updated
[ ] Module's ReadMe.LLM updated if public API changed
```

---
"""

    # ---- key modules & entry points ----

    def _key_modules_section(self) -> str:
        if not self.key_modules:
            return ""

        lines = ["## Key Modules\n"]
        lines.append("| Module | Description |")
        lines.append("|--------|-------------|")
        for mod in self.key_modules[:10]:
            if isinstance(mod, dict):
                name = mod.get("name", "Unknown")
                desc = mod.get("description", "N/A")
            else:
                name = str(mod)
                desc = "See `ReadMe.LLM` inside"
            lines.append(f"| `{name}` | {desc} |")
        lines.append("\n---\n")
        return "\n".join(lines)

    def _entry_points_section(self) -> str:
        if not self.entry_points:
            return ""

        lines = ["## Entry Points\n"]
        for ep in self.entry_points[:10]:
            lines.append(f"- `{ep}`")
        lines.append("\n---\n")
        return "\n".join(lines)

    def _footer(self) -> str:
        return f"\n*Generated by Code-In Agent — {self.date}*\n"


# ═══════════════════════════════════════════════════════════════
#  repomap.txt  —  AST-Compressed Repository Map
# ═══════════════════════════════════════════════════════════════

class RepomapGenerator:
    """
    Generates ``repomap.txt`` — a compressed, AST-based map of the
    entire repository showing only high-level symbols.

    The output is a directory tree annotated with:
    - Class names and their public methods
    - Top-level function signatures
    - Exported symbols

    No function bodies are included.  This fits in a single context
    window for most medium-sized projects.
    """

    def __init__(
        self,
        project_name: str,
        file_tree: list,
        dependency_graph: dict,
        main_language: str,
        profile: ProjectProfile,
    ):
        self.project_name = project_name
        self.file_tree = file_tree or []
        self.dep_graph = dependency_graph or {}
        self.language = main_language
        self.profile = profile
        self.date = datetime.now().strftime("%Y-%m-%d")

    def generate(self) -> str:
        """Return the full content of ``repomap.txt``."""
        sections = [
            self._header(),
            self._stats(),
            self._tree(),
        ]
        return "\n".join(sections)

    # ---- header ----

    def _header(self) -> str:
        return f"""# repomap.txt — {self.project_name}
# AST-compressed repository map (symbols only, no bodies)
# Generated: {self.date}
# Language: {self.language}
#
"""

    def _stats(self) -> str:
        stats = self.dep_graph.get("stats", {})
        return f"""# Stats: {stats.get('file_count', self.profile.total_files)} files, \
{stats.get('function_count', self.profile.total_functions)} functions, \
{stats.get('class_count', self.profile.total_classes)} classes
#
"""

    # ---- main tree ----

    def _tree(self) -> str:
        """Build the annotated directory tree."""
        # Group files by directory
        dir_map: dict[str, list[dict]] = {}
        root_files: list[dict] = []

        for f in self.file_tree:
            path = f.get("path", "")
            parent = str(Path(path).parent)

            # Skip noise
            if any(
                skip in path
                for skip in (
                    "node_modules", "__pycache__", ".git", "venv",
                    "dist/", "build/", ".next/",
                )
            ):
                continue

            if parent == "." or parent == "":
                root_files.append(f)
            else:
                dir_map.setdefault(parent, []).append(f)

        lines: list[str] = []

        # Root files first
        for f in sorted(root_files, key=lambda x: x.get("path", "")):
            lines.extend(self._format_file(f, indent=0))

        # Then directories sorted alphabetically
        for dir_path in sorted(dir_map.keys()):
            files = dir_map[dir_path]
            lines.append(f"\n{dir_path}/")
            for f in sorted(files, key=lambda x: x.get("path", "")):
                lines.extend(self._format_file(f, indent=1))

        return "\n".join(lines)

    def _format_file(self, file_info: dict, indent: int = 0) -> list[str]:
        """Format a single file with its high-level symbols."""
        path = file_info.get("path", "")
        name = Path(path).name
        prefix = "  " * indent
        lines: list[str] = []

        # File line with line count
        lc = file_info.get("line_count", 0)
        lc_str = f"  ({lc}L)" if lc else ""
        lines.append(f"{prefix}├── {name}{lc_str}")

        inner_prefix = prefix + "│   "

        # Classes
        for cls in file_info.get("class_details", []):
            if not isinstance(cls, dict):
                continue
            cls_name = cls.get("name", "?")
            bases = cls.get("bases", [])
            base_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"{inner_prefix}class {cls_name}{base_str}")

            # Public methods only
            for method in cls.get("method_details", cls.get("methods", [])):
                if isinstance(method, dict):
                    m_name = method.get("name", "?")
                    if m_name.startswith("_") and not m_name.startswith("__"):
                        continue  # skip private
                    params = method.get("parameters", [])
                    ret = method.get("return_type", "")
                    sig = f"{m_name}({', '.join(params[:4])})"
                    if ret:
                        sig += f" -> {ret}"
                    is_async = method.get("is_async", False)
                    prefix_kw = "async " if is_async else ""
                    lines.append(f"{inner_prefix}  {prefix_kw}def {sig}")
                elif isinstance(method, str):
                    lines.append(f"{inner_prefix}  def {method}()")

        # Top-level functions (non-methods)
        for func in file_info.get("function_details", []):
            if not isinstance(func, dict):
                continue
            if func.get("is_method", False):
                continue
            f_name = func.get("name", "?")
            if f_name.startswith("_") and not f_name.startswith("__"):
                continue
            params = func.get("parameters", [])
            ret = func.get("return_type", "")
            sig = f"{f_name}({', '.join(params[:4])})"
            if ret:
                sig += f" -> {ret}"
            is_async = func.get("is_async", False)
            prefix_kw = "async " if is_async else ""
            lines.append(f"{inner_prefix}{prefix_kw}def {sig}")

        # Exports (if JS/TS)
        exports = file_info.get("exports", [])
        if exports:
            exp_str = ", ".join(str(e) for e in exports[:8])
            if len(exports) > 8:
                exp_str += f" (+{len(exports) - 8})"
            lines.append(f"{inner_prefix}exports: {exp_str}")

        return lines
