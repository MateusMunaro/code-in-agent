"""
Module Level Output Generators.

Generates per-module (per-subdirectory) documentation artifacts:
- {module}/ReadMe.LLM   — Function signatures + I/O examples
- {module}/AGENTS.md     — Module-specific rule overrides

These files live inside each code subdirectory and provide focused
context for AI agents operating within a specific module.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from .mermaid_generator import MermaidGenerator, MermaidContext


# ═══════════════════════════════════════════════════════════════
#  ReadMe.LLM  —  Module Signatures & I/O
# ═══════════════════════════════════════════════════════════════

class ReadmeLlmGenerator:
    """
    Generates ``ReadMe.LLM`` for a single module directory.

    Content:
    - Module summary (files, functions, classes)
    - Function / method signatures with types
    - Class hierarchy within the module
    - I/O examples inferred from test files
    - Internal imports and dependencies
    - Intra-module Mermaid diagram
    """

    def __init__(
        self,
        module_path: str,
        module_files: list[dict],
        test_files: list[dict],
        dependency_graph: dict,
        main_language: str,
        project_name: str,
    ):
        self.module_path = module_path
        self.files = module_files
        self.tests = test_files
        self.dep_graph = dependency_graph or {}
        self.language = main_language
        self.project_name = project_name
        self.date = datetime.now().strftime("%Y-%m-%d")

    def generate(self) -> str:
        sections = [
            self._header(),
            self._summary(),
            self._signatures(),
            self._io_examples(),
            self._dependencies(),
            self._diagram(),
            self._footer(),
        ]
        return "\n".join(s for s in sections if s)

    # ---- header ----

    def _header(self) -> str:
        return f"""# ReadMe.LLM — {self.module_path}/

> Machine-readable module documentation for `{self.project_name}`
> Generated: {self.date}

---
"""

    # ---- summary ----

    def _summary(self) -> str:
        total_funcs = sum(
            len(f.get("function_details", [])) for f in self.files
        )
        total_classes = sum(
            len(f.get("class_details", [])) for f in self.files
        )
        file_names = [Path(f.get("path", "")).name for f in self.files]

        lines = [
            "## Module Summary\n",
            f"- **Files:** {len(self.files)}",
            f"- **Functions:** {total_funcs}",
            f"- **Classes:** {total_classes}",
            f"- **Language:** {self.language}",
            "",
            "### Files",
            "",
        ]
        for name in sorted(file_names):
            lines.append(f"- `{name}`")

        lines.append("\n---\n")
        return "\n".join(lines)

    # ---- signatures ----

    def _signatures(self) -> str:
        lines = ["## Signatures\n"]

        for f_info in sorted(self.files, key=lambda x: x.get("path", "")):
            file_name = Path(f_info.get("path", "")).name
            classes = f_info.get("class_details", [])
            funcs = f_info.get("function_details", [])

            if not classes and not funcs:
                continue

            lines.append(f"### `{file_name}`\n")

            # Classes
            for cls in classes:
                if not isinstance(cls, dict):
                    continue
                cls_name = cls.get("name", "?")
                bases = cls.get("bases", [])
                base_str = f"({', '.join(bases)})" if bases else ""
                docstring = cls.get("docstring", "")

                lines.append(f"**class `{cls_name}{base_str}`**")
                if docstring:
                    lines.append(f"> {docstring[:200]}")
                lines.append("")

                # Methods
                method_details = cls.get("method_details", [])
                if method_details:
                    lines.append("| Method | Params | Returns | Async |")
                    lines.append("|--------|--------|---------|-------|")
                    for m in method_details:
                        if not isinstance(m, dict):
                            continue
                        m_name = m.get("name", "?")
                        params = ", ".join(m.get("parameters", [])[:5])
                        ret = m.get("return_type", "-")
                        is_async = "✓" if m.get("is_async") else ""
                        lines.append(f"| `{m_name}` | `{params}` | `{ret}` | {is_async} |")
                    lines.append("")
                elif cls.get("methods"):
                    # Fallback: method names only
                    lines.append("Methods: " + ", ".join(
                        f"`{m}`" for m in cls["methods"][:10]
                    ))
                    lines.append("")

            # Top-level functions
            top_funcs = [
                fn for fn in funcs
                if isinstance(fn, dict) and not fn.get("is_method", False)
            ]
            if top_funcs:
                lines.append("| Function | Params | Returns | Async |")
                lines.append("|----------|--------|---------|-------|")
                for fn in top_funcs:
                    name = fn.get("name", "?")
                    params = ", ".join(fn.get("parameters", [])[:5])
                    ret = fn.get("return_type", "-")
                    is_async = "✓" if fn.get("is_async") else ""
                    lines.append(f"| `{name}` | `{params}` | `{ret}` | {is_async} |")
                lines.append("")

        lines.append("---\n")
        return "\n".join(lines)

    # ---- I/O from tests ----

    def _io_examples(self) -> str:
        if not self.tests:
            return ""

        lines = ["## I/O Examples (from tests)\n"]

        for test_file in self.tests[:5]:
            test_name = Path(test_file.get("path", "")).name
            lines.append(f"### `{test_name}`\n")

            for fn in test_file.get("function_details", []):
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name", "")
                docstring = fn.get("docstring", "")
                if name.startswith("test_") or name.startswith("test"):
                    lines.append(f"- **`{name}`**")
                    if docstring:
                        lines.append(f"  > {docstring[:150]}")

            lines.append("")

        lines.append("---\n")
        return "\n".join(lines)

    # ---- dependencies ----

    def _dependencies(self) -> str:
        edges = self.dep_graph.get("edges", [])
        module_file_paths = {f.get("path", "") for f in self.files}

        # Outgoing imports (this module → others)
        outgoing: dict[str, set[str]] = {}
        # Incoming imports (others → this module)
        incoming: dict[str, set[str]] = {}

        for edge in edges:
            if edge.get("type") != "imports":
                continue
            src = edge.get("source", "").replace("file:", "")
            tgt = edge.get("target", "").replace("file:", "")

            if src in module_file_paths and tgt not in module_file_paths:
                outgoing.setdefault(Path(src).name, set()).add(tgt)
            elif tgt in module_file_paths and src not in module_file_paths:
                incoming.setdefault(Path(tgt).name, set()).add(src)

        if not outgoing and not incoming:
            return ""

        lines = ["## Dependencies\n"]

        if outgoing:
            lines.append("### This module imports from:")
            for src_name, targets in sorted(outgoing.items()):
                for t in sorted(targets):
                    lines.append(f"- `{src_name}` → `{t}`")
            lines.append("")

        if incoming:
            lines.append("### Imported by:")
            for tgt_name, sources in sorted(incoming.items()):
                for s in sorted(sources):
                    lines.append(f"- `{s}` → `{tgt_name}`")
            lines.append("")

        lines.append("---\n")
        return "\n".join(lines)

    # ---- mermaid diagram ----

    def _diagram(self) -> str:
        if len(self.files) < 2:
            return ""

        ctx = MermaidContext(
            architecture_pattern="",
            file_tree=self.files,
            dependency_graph=self.dep_graph,
            entry_points=[],
            key_modules=[],
            main_language=self.language,
        )
        gen = MermaidGenerator(ctx)
        diagram = gen.module_diagram(self.module_path, self.files)

        if not diagram:
            return ""

        return f"""## Module Structure

```mermaid
{diagram}
```

---
"""

    def _footer(self) -> str:
        return f"\n*Generated by Code-In Agent — {self.date}*\n"


# ═══════════════════════════════════════════════════════════════
#  Nested AGENTS.md  —  Module-Specific Overrides
# ═══════════════════════════════════════════════════════════════

class NestedAgentsMdGenerator:
    """
    Generates a module-level ``AGENTS.md`` with overrides specific
    to the type of code found in the directory.

    Detects module type and generates appropriate rules:
    - tests/     → "use existing fixtures", "follow test patterns"
    - routes/    → "follow REST conventions", "validate inputs"
    - components/ → "use design system", "handle loading states"
    - services/  → "follow repository pattern", "handle errors"
    - models/    → "validate schemas", "use migrations"
    """

    # Module type detection heuristics
    TYPE_PATTERNS = {
        "tests": {
            "path_keywords": ["test", "tests", "spec", "specs", "__tests__"],
            "file_keywords": ["test_", "_test.", ".spec.", ".test."],
        },
        "api": {
            "path_keywords": ["routes", "api", "handlers", "controllers", "endpoints"],
            "file_keywords": ["route", "handler", "controller", "endpoint"],
        },
        "components": {
            "path_keywords": ["components", "ui", "widgets", "views"],
            "file_keywords": [".tsx", ".jsx", "component"],
        },
        "services": {
            "path_keywords": ["services", "service", "providers", "usecases"],
            "file_keywords": ["service", "provider", "usecase"],
        },
        "models": {
            "path_keywords": ["models", "entities", "domain", "schemas"],
            "file_keywords": ["model", "entity", "schema"],
        },
        "config": {
            "path_keywords": ["config", "configuration", "settings"],
            "file_keywords": ["config", "settings"],
        },
        "utils": {
            "path_keywords": ["utils", "helpers", "lib", "shared", "common"],
            "file_keywords": ["util", "helper", "common"],
        },
    }

    def __init__(
        self,
        module_path: str,
        module_files: list[dict],
        main_language: str,
        project_name: str,
        architecture_pattern: str,
    ):
        self.module_path = module_path
        self.files = module_files
        self.language = main_language
        self.project_name = project_name
        self.architecture = architecture_pattern
        self.module_type = self._detect_module_type()
        self.date = datetime.now().strftime("%B %Y")

    def generate(self) -> str:
        sections = [
            self._header(),
            self._module_rules(),
            self._file_list(),
            self._footer(),
        ]
        return "\n".join(s for s in sections if s)

    # ---- type detection ----

    def _detect_module_type(self) -> str:
        """Detect the module type from path and file patterns."""
        path_lower = self.module_path.lower()
        file_paths = [f.get("path", "").lower() for f in self.files]

        for mod_type, patterns in self.TYPE_PATTERNS.items():
            # Check directory path
            for kw in patterns["path_keywords"]:
                if kw in path_lower:
                    return mod_type
            # Check file names
            for fp in file_paths:
                for kw in patterns["file_keywords"]:
                    if kw in fp:
                        return mod_type

        return "generic"

    # ---- content ----

    def _header(self) -> str:
        type_label = self.module_type.title()
        return f"""# AGENTS.md — {self.module_path}/

> Module-specific overrides for `{self.project_name}`
> Module type: **{type_label}**
> These rules **extend** (not replace) the root AGENTS.md

---
"""

    def _module_rules(self) -> str:
        rules_map = {
            "tests": self._rules_tests,
            "api": self._rules_api,
            "components": self._rules_components,
            "services": self._rules_services,
            "models": self._rules_models,
            "config": self._rules_config,
            "utils": self._rules_utils,
            "generic": self._rules_generic,
        }
        generator = rules_map.get(self.module_type, self._rules_generic)
        return generator()

    def _rules_tests(self) -> str:
        return """## Test Module Rules

1. **Use existing fixtures** — check for `conftest.py`, `fixtures/`, or setup files before creating new ones
2. **Follow test naming** — prefix test functions with `test_` (Python) or `describe`/`it` blocks (JS/TS)
3. **One assertion per concept** — test a single behavior per test function
4. **Don't test implementation details** — test behavior and public APIs
5. **Mock external dependencies** — never make real HTTP/DB calls in unit tests
6. **Keep tests isolated** — tests must not depend on execution order

### Test File Naming
```
test_{module_name}.py          # Python
{module_name}.test.ts          # TypeScript
{module_name}.spec.ts          # TypeScript (alt)
```

---
"""

    def _rules_api(self) -> str:
        return """## API / Routes Module Rules

1. **Follow REST conventions** — use proper HTTP methods (GET, POST, PUT, DELETE)
2. **Validate all inputs** — never trust client data; validate at the boundary
3. **Consistent error responses** — use the existing error format, do not invent new ones
4. **Document endpoints** — include request/response types
5. **Handle authentication** — check existing auth middleware before adding new patterns
6. **Pagination** — use the existing pagination pattern for list endpoints

### Route Pattern
```
GET    /resource          → list
GET    /resource/:id      → get one
POST   /resource          → create
PUT    /resource/:id      → update
DELETE /resource/:id      → delete
```

---
"""

    def _rules_components(self) -> str:
        return """## UI Components Module Rules

1. **Use the design system** — check existing components before creating new ones
2. **Handle all states** — loading, error, empty, and success states
3. **Props typing** — define explicit prop interfaces/types
4. **Accessibility** — use semantic HTML, ARIA labels where needed
5. **Responsive** — follow the existing responsive patterns
6. **Composition over inheritance** — prefer composition and hooks

### Component Structure
```
ComponentName/
├── index.ts              (re-export)
├── ComponentName.tsx     (main component)
├── ComponentName.test.tsx
└── types.ts              (if complex types)
```

---
"""

    def _rules_services(self) -> str:
        return """## Services Module Rules

1. **Single responsibility** — one service, one domain concern
2. **Error handling** — wrap external calls in try/catch, return typed errors
3. **Dependency injection** — receive dependencies via constructor / parameters
4. **Logging** — log entry, exit, and errors for external service calls
5. **Idempotency** — design write operations to be safely retried
6. **Interface first** — define the service interface before implementing

---
"""

    def _rules_models(self) -> str:
        return """## Models / Domain Module Rules

1. **Validate schemas** — add validation to data classes / interfaces
2. **Use migrations** — never modify the database schema manually
3. **Immutable where possible** — prefer immutable data structures
4. **No side effects** — domain logic must be pure (no I/O)
5. **Type everything** — all fields must have explicit types
6. **Document invariants** — describe business rules as comments or docstrings

---
"""

    def _rules_config(self) -> str:
        return """## Configuration Module Rules

1. **Environment variables** — sensitive values come from env vars, never hardcoded
2. **Defaults** — always provide sensible defaults
3. **Validation** — validate config at startup, fail fast on invalid values
4. **Type safety** — use typed config classes / schemas
5. **No business logic** — this module only holds configuration, not behavior

---
"""

    def _rules_utils(self) -> str:
        return """## Utils / Helpers Module Rules

1. **Pure functions** — utilities should be pure, no hidden state
2. **Generic** — do not put domain-specific code here
3. **Tested** — every utility function must have unit tests
4. **Documented** — include docstring/JSDoc with params, returns, and examples
5. **No external dependencies** — minimize third-party imports in utility code

---
"""

    def _rules_generic(self) -> str:
        return """## Module Rules

1. Follow the patterns established in this directory
2. Check existing files before adding new ones
3. Maintain consistent naming with sibling files
4. Update this module's `ReadMe.LLM` when adding new public APIs

---
"""

    def _file_list(self) -> str:
        if not self.files:
            return ""

        lines = ["## Files in This Module\n"]
        for f in sorted(self.files, key=lambda x: x.get("path", "")):
            name = Path(f.get("path", "")).name
            funcs = len(f.get("function_details", []))
            classes = len(f.get("class_details", []))
            parts = []
            if funcs:
                parts.append(f"{funcs} functions")
            if classes:
                parts.append(f"{classes} classes")
            desc = f" — {', '.join(parts)}" if parts else ""
            lines.append(f"- `{name}`{desc}")

        lines.append("\n---\n")
        return "\n".join(lines)

    def _footer(self) -> str:
        return f"\n*Generated by Code-In Agent — {self.date}*\n"
