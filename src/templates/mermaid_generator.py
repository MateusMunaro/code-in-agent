"""
Mermaid Diagram Generator.

Extracted from DocumentationGenerator for reuse across all documentation layers:
Root Level (AGENTS.md), Module Level (ReadMe.LLM), and Semantic Layer.

Generates Mermaid diagrams from real AST data and dependency graphs.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class MermaidContext:
    """Context needed to generate Mermaid diagrams."""
    architecture_pattern: str
    file_tree: list
    dependency_graph: dict
    entry_points: list
    key_modules: list
    main_language: str
    framework: Optional[str] = None


class MermaidGenerator:
    """
    Generates Mermaid diagrams from real code analysis data.
    
    Supports:
    - Architecture diagrams (pattern-aware)
    - Layer diagrams (from directory structure)
    - Class diagrams (from AST)
    - Sequence diagrams (from call graph)
    - Component diagrams (from module structure)
    - Data flow diagrams (from entry points + deps)
    - Dependency graphs (from import/call edges)
    """

    def __init__(self, ctx: MermaidContext):
        self.ctx = ctx

    # =========================================
    # PUBLIC API
    # =========================================

    def architecture(self) -> str:
        """Generate architecture Mermaid diagram based on detected pattern."""
        arch = self.ctx.architecture_pattern.lower()

        if "clean" in arch:
            return self._arch_clean()
        elif "mvc" in arch:
            return self._arch_mvc()
        elif "hexagonal" in arch or "ports" in arch:
            return self._arch_hexagonal()
        else:
            return self._arch_generic()

    def layers(self) -> str:
        """Generate layers diagram from actual directory structure."""
        file_tree = self.ctx.file_tree or []
        if not file_tree:
            return self._layers_fallback()

        dir_groups = {}
        for f in file_tree:
            path = f.get("path", "")
            parts = Path(path).parts
            if len(parts) > 1:
                top_dir = parts[0]
                if top_dir.startswith(".") or top_dir in (
                    "node_modules", "__pycache__", ".git", "venv", "dist", "build"
                ):
                    continue
                if top_dir not in dir_groups:
                    dir_groups[top_dir] = {"files": 0, "funcs": 0, "classes": 0}
                dir_groups[top_dir]["files"] += 1
                dir_groups[top_dir]["funcs"] += len(f.get("function_details", []))
                dir_groups[top_dir]["classes"] += len(f.get("class_details", []))

        if not dir_groups:
            return self._layers_fallback()

        lines = ["flowchart TB"]
        sorted_dirs = sorted(
            dir_groups.items(), key=lambda x: x[1]["files"], reverse=True
        )[:8]

        for _i, (dir_name, stats) in enumerate(sorted_dirs):
            safe = dir_name.replace("-", "_").replace(".", "_")
            label = f"{dir_name} ({stats['files']} files"
            if stats["funcs"]:
                label += f", {stats['funcs']} funcs"
            if stats["classes"]:
                label += f", {stats['classes']} classes"
            label += ")"
            lines.append(f'    {safe}["{label}"]')

        # Add edges based on dependency graph imports between directories
        edges = self.ctx.dependency_graph.get("edges", [])
        dir_edges: set[tuple[str, str]] = set()
        valid_ids = {
            d[0].replace("-", "_").replace(".", "_") for d in sorted_dirs
        }
        for edge in edges:
            if edge.get("type") == "imports":
                src = edge.get("source", "").replace("file:", "")
                tgt = edge.get("target", "").replace("file:", "")
                src_dir = Path(src).parts[0] if Path(src).parts else ""
                tgt_dir = Path(tgt).parts[0] if Path(tgt).parts else ""
                if src_dir and tgt_dir and src_dir != tgt_dir:
                    s1 = src_dir.replace("-", "_").replace(".", "_")
                    s2 = tgt_dir.replace("-", "_").replace(".", "_")
                    if s1 in valid_ids and s2 in valid_ids:
                        dir_edges.add((s1, s2))

        for src, tgt in list(dir_edges)[:15]:
            lines.append(f"    {src} --> {tgt}")

        return "\n".join(lines)

    def class_diagram(self) -> str:
        """Generate class diagram from actual code analysis."""
        nodes = self.ctx.dependency_graph.get("nodes", [])
        edges = self.ctx.dependency_graph.get("edges", [])
        classes = [n for n in nodes if n.get("type") == "class"]

        if not classes:
            return self._class_fallback()

        lines = ["classDiagram"]

        for cls in classes[:15]:
            name = cls.get("name", "Unknown")
            metadata = cls.get("metadata", {})
            methods = metadata.get("methods", [])
            safe_name = name.replace("-", "_").replace(".", "_")

            lines.append(f"    class {safe_name} {{")
            for method in methods[:5]:
                lines.append(f"        +{method}()")
            lines.append("    }")

        for edge in edges:
            if edge.get("type") == "extends":
                source_name = edge.get("source", "").split(":")[-1]
                target_name = edge.get("target", "").split(":")[-1]
                s_safe = source_name.replace("-", "_").replace(".", "_")
                t_safe = target_name.replace("-", "_").replace(".", "_")
                if s_safe and t_safe:
                    lines.append(f"    {t_safe} <|-- {s_safe}")

        for edge in edges:
            if edge.get("type") == "implements":
                source_name = edge.get("source", "").split(":")[-1]
                target_name = edge.get("target", "").split(":")[-1]
                s_safe = source_name.replace("-", "_").replace(".", "_")
                t_safe = target_name.replace("-", "_").replace(".", "_")
                if s_safe and t_safe:
                    lines.append(f"    {t_safe} <|.. {s_safe} : implements")

        return "\n".join(lines)

    def sequence(self) -> str:
        """Generate sequence diagram from actual call graph."""
        edges = self.ctx.dependency_graph.get("edges", [])
        call_edges = [e for e in edges if e.get("type") == "calls"]

        if not call_edges:
            import_edges = [e for e in edges if e.get("type") == "imports"][:6]
            if not import_edges:
                return self._sequence_fallback()

            lines = ["sequenceDiagram"]
            seen: set[tuple[str, str]] = set()
            for edge in import_edges:
                src = Path(edge.get("source", "").replace("file:", "")).stem[:20]
                tgt = Path(edge.get("target", "").replace("file:", "")).stem[:20]
                if src and tgt and src != tgt and (src, tgt) not in seen:
                    seen.add((src, tgt))
                    lines.append(f"    {src}->>+{tgt}: imports")
                    lines.append(f"    {tgt}-->>-{src}: provides")
            return "\n".join(lines) if len(lines) > 1 else self._sequence_fallback()

        lines = ["sequenceDiagram"]
        participants: set[str] = set()
        seen_calls: set[tuple[str, str]] = set()

        for edge in call_edges[:10]:
            caller = edge.get("source", "").split(":")[-1][:20]
            callee = edge.get("target", "").split(":")[-1][:20]
            if not caller or not callee or caller == callee:
                continue

            caller_safe = caller.replace(".", "_").replace("-", "_").replace(" ", "_")
            callee_safe = callee.replace(".", "_").replace("-", "_").replace(" ", "_")

            if caller_safe not in participants:
                lines.append(f"    participant {caller_safe} as {caller}")
                participants.add(caller_safe)
            if callee_safe not in participants:
                lines.append(f"    participant {callee_safe} as {callee}")
                participants.add(callee_safe)

            call_key = (caller_safe, callee_safe)
            if call_key not in seen_calls:
                seen_calls.add(call_key)
                lines.append(f"    {caller_safe}->>+{callee_safe}: call")
                lines.append(f"    {callee_safe}-->>-{caller_safe}: return")

        return "\n".join(lines) if len(lines) > 1 else self._sequence_fallback()

    def component(self) -> str:
        """Generate component diagram from module structure."""
        modules = self.ctx.key_modules[:8] if self.ctx.key_modules else []

        if not modules:
            return self._component_fallback()

        lines = ["flowchart TB"]
        for i, module in enumerate(modules):
            name = module if isinstance(module, str) else module.get("name", f"Module{i}")
            clean = name.replace("/", "_").replace("-", "_").replace(".", "_")
            lines.append(f"    {clean}[{name}]")

        return "\n".join(lines)

    def data_flow(self) -> str:
        """Generate data flow diagram from entry points and call chains."""
        entry_points = self.ctx.entry_points or []
        edges = self.ctx.dependency_graph.get("edges", [])

        if not entry_points and not edges:
            return self._data_flow_fallback()

        lines = ["flowchart LR"]

        if entry_points:
            lines.append('    subgraph Entry["📥 Entry Points"]')
            for i, ep in enumerate(entry_points[:5]):
                name = Path(ep).stem if isinstance(ep, str) else str(ep)
                safe = f"ep_{name.replace('.', '_').replace('-', '_').replace(' ', '_')[:15]}"
                lines.append(f'        {safe}["{name}"]')
            lines.append("    end")

        import_targets: dict[str, int] = {}
        for edge in edges:
            if edge.get("type") == "imports":
                tgt = edge.get("target", "").replace("file:", "")
                tgt_name = Path(tgt).stem
                import_targets[tgt_name] = import_targets.get(tgt_name, 0) + 1

        core_modules = sorted(
            import_targets.items(), key=lambda x: x[1], reverse=True
        )[:5]
        if core_modules:
            lines.append('    subgraph Core["⚙️ Core Modules"]')
            for name, count in core_modules:
                safe = f"core_{name.replace('.', '_').replace('-', '_')[:15]}"
                lines.append(f'        {safe}["{name} ({count}x imported)"]')
            lines.append("    end")

        if entry_points and core_modules:
            ep_name = Path(entry_points[0]).stem if isinstance(entry_points[0], str) else "entry"
            ep_safe = f"ep_{ep_name.replace('.', '_').replace('-', '_').replace(' ', '_')[:15]}"
            core_safe = f"core_{core_modules[0][0].replace('.', '_').replace('-', '_')[:15]}"
            lines.append(f"    {ep_safe} --> {core_safe}")

        return "\n".join(lines) if len(lines) > 1 else self._data_flow_fallback()

    def dependency_graph(self) -> str:
        """Generate dependency graph Mermaid from actual code relationships."""
        edges = self.ctx.dependency_graph.get("edges", [])

        if not edges:
            return self._dependency_fallback()

        lines = ["flowchart TB"]
        seen_nodes: set[str] = set()

        import_edges = [e for e in edges if e.get("type") == "imports"][:15]
        call_edges = [e for e in edges if e.get("type") == "calls"][:15]

        if import_edges:
            lines.append("    subgraph Imports[File Dependencies]")
            for edge in import_edges:
                source = edge.get("source", "").replace("file:", "").split("/")[-1]
                target = edge.get("target", "").replace("file:", "").split("/")[-1]
                if source and target and source != target:
                    sc = source.replace(".", "_").replace("-", "_")[:20]
                    tc = target.replace(".", "_").replace("-", "_")[:20]
                    if sc not in seen_nodes:
                        lines.append(f"        {sc}[{source[:15]}]")
                        seen_nodes.add(sc)
                    if tc not in seen_nodes:
                        lines.append(f"        {tc}[{target[:15]}]")
                        seen_nodes.add(tc)
                    lines.append(f"        {sc} --> {tc}")
            lines.append("    end")

        if call_edges:
            lines.append("    subgraph Calls[Function Calls]")
            call_nodes: set[str] = set()
            for edge in call_edges:
                source = edge.get("source", "").split(":")[-1]
                target = edge.get("target", "").split(":")[-1]
                if source and target and source != target:
                    sc = f"fn_{source.replace('.', '_').replace('-', '_')[:15]}"
                    tc = f"fn_{target.replace('.', '_').replace('-', '_')[:15]}"
                    if sc not in call_nodes:
                        lines.append(f"        {sc}({source[:12]})")
                        call_nodes.add(sc)
                    if tc not in call_nodes:
                        lines.append(f"        {tc}({target[:12]})")
                        call_nodes.add(tc)
                    lines.append(f"        {sc} -.-> {tc}")
            lines.append("    end")

        return "\n".join(lines) if len(lines) > 1 else self._dependency_fallback()

    def module_diagram(self, module_path: str, module_files: list[dict]) -> str:
        """
        Generate a focused diagram for a specific module's internal structure.

        Args:
            module_path: The module directory path (e.g. "src/services")
            module_files: List of FileInfo dicts belonging to this module
        """
        if not module_files:
            return ""

        lines = ["flowchart TB"]
        safe_module = module_path.replace("/", "_").replace("-", "_").replace(".", "_")
        lines.append(f'    subgraph {safe_module}["{module_path}/"]')

        file_ids: dict[str, str] = {}
        for f in module_files[:12]:
            name = Path(f.get("path", "")).name
            safe = name.replace(".", "_").replace("-", "_")[:20]
            uid = f"{safe_module}_{safe}"
            funcs = len(f.get("function_details", []))
            classes = len(f.get("class_details", []))
            label = name
            if funcs or classes:
                parts = []
                if funcs:
                    parts.append(f"{funcs}F")
                if classes:
                    parts.append(f"{classes}C")
                label += f" ({', '.join(parts)})"
            lines.append(f'        {uid}["{label}"]')
            file_ids[f.get("path", "")] = uid

        # Internal edges
        dep_edges = self.ctx.dependency_graph.get("edges", [])
        for edge in dep_edges:
            if edge.get("type") == "imports":
                src_path = edge.get("source", "").replace("file:", "")
                tgt_path = edge.get("target", "").replace("file:", "")
                if src_path in file_ids and tgt_path in file_ids:
                    lines.append(f"        {file_ids[src_path]} --> {file_ids[tgt_path]}")

        lines.append("    end")
        return "\n".join(lines)

    # =========================================
    # FALLBACKS
    # =========================================

    def _layers_fallback(self) -> str:
        return 'flowchart TB\n    Source["Source Files"]\n    Config["Configuration"]\n    Source --> Config'

    def _class_fallback(self) -> str:
        return (
            "classDiagram\n"
            "    class Entity {\n        +id: string\n        +createdAt: datetime\n    }\n"
            "    class Service {\n        +execute()\n    }\n"
            "    Entity <-- Service : uses"
        )

    def _sequence_fallback(self) -> str:
        return (
            "sequenceDiagram\n"
            "    participant Entry as Entry Point\n"
            "    participant Core as Core Logic\n"
            "    Entry->>+Core: process\n"
            "    Core-->>-Entry: result"
        )

    def _component_fallback(self) -> str:
        return (
            "flowchart TB\n"
            '    subgraph Core["Core"]\n        Main[Main Module]\n    end\n'
            '    subgraph Features["Features"]\n        F1[Feature 1]\n        F2[Feature 2]\n    end\n'
            "    Main --> F1\n    Main --> F2"
        )

    def _data_flow_fallback(self) -> str:
        return 'flowchart LR\n    Input["📥 Input"] --> Process["⚙️ Processing"] --> Output["📤 Output"]'

    def _dependency_fallback(self) -> str:
        return "flowchart TB\n    A[Module A] --> B[Module B]\n    B --> C[Module C]"

    # =========================================
    # ARCHITECTURE TEMPLATES
    # =========================================

    def _arch_clean(self) -> str:
        return """flowchart TB
    subgraph External["🌐 External"]
        UI[UI/API]
        DB[(Database)]
        ExtAPI[External APIs]
    end
    
    subgraph Adapters["📦 Adapters"]
        Controllers[Controllers]
        Repositories[Repositories]
        Gateways[Gateways]
    end
    
    subgraph Application["⚙️ Application"]
        UseCases[Use Cases]
        Services[Services]
    end
    
    subgraph Domain["💎 Domain"]
        Entities[Entities]
        ValueObjects[Value Objects]
    end
    
    UI --> Controllers
    Controllers --> UseCases
    UseCases --> Entities
    UseCases --> Repositories
    Repositories --> DB
    UseCases --> Gateways
    Gateways --> ExtAPI"""

    def _arch_mvc(self) -> str:
        return """flowchart LR
    subgraph View["👁️ View"]
        Templates[Templates]
        Components[Components]
    end
    
    subgraph Controller["🎮 Controller"]
        Routes[Routes]
        Handlers[Handlers]
    end
    
    subgraph Model["📦 Model"]
        Entities[Entities]
        Repositories[Repositories]
    end
    
    View <--> Controller
    Controller <--> Model"""

    def _arch_hexagonal(self) -> str:
        return """flowchart TB
    subgraph Inbound["⬅️ Inbound Adapters"]
        HTTP[HTTP API]
        CLI[CLI]
        Events[Event Listeners]
    end
    
    subgraph Core["💎 Core Domain"]
        subgraph Ports["Ports"]
            InPorts[Inbound Ports]
            OutPorts[Outbound Ports]
        end
        Domain[Domain Logic]
    end
    
    subgraph Outbound["➡️ Outbound Adapters"]
        DB[(Database)]
        ExtAPI[External APIs]
        Queue[Message Queue]
    end
    
    HTTP --> InPorts
    CLI --> InPorts
    Events --> InPorts
    InPorts --> Domain
    Domain --> OutPorts
    OutPorts --> DB
    OutPorts --> ExtAPI
    OutPorts --> Queue"""

    def _arch_generic(self) -> str:
        return """flowchart TB
    subgraph Presentation["🖥️ Presentation"]
        UI[User Interface]
        API[API Layer]
    end
    
    subgraph Business["⚙️ Business Logic"]
        Services[Services]
        Logic[Core Logic]
    end
    
    subgraph Data["🗄️ Data Layer"]
        DB[(Database)]
        Cache[Cache]
    end
    
    UI --> Services
    API --> Services
    Services --> Logic
    Logic --> DB
    Logic --> Cache"""
