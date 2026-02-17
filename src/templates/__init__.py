"""
Multi-layer documentation templates for AI-friendly codebase documentation.

Architecture:
    Root Level   → llms.txt, AGENTS.md, repomap.txt
    Module Level → {module}/ReadMe.LLM, {module}/AGENTS.md
    Semantic     → _codein/ (SCIP, TreeFrag, KG) — Phase 3
"""

# ── New multi-layer generators ──────────────────────────────
from .root_level import (
    LlmsTxtGenerator,
    AgentsMdGenerator,
    RepomapGenerator,
    ProjectProfile,
)
from .module_level import ReadmeLlmGenerator, NestedAgentsMdGenerator
from .mermaid_generator import MermaidGenerator, MermaidContext

# ── Main orchestrator ───────────────────────────────────────
from .doc_generator import (
    MultiLayerDocGenerator,
    DocumentationGenerator,  # backward-compat alias
    AnalysisResult,
    generate_documentation,
)

# ── Legacy wrappers (backward-compat) ──────────────────────
from .agent_guidelines import (
    AgentGuidelinesGenerator,
    generate_agent_guidelines,
    ProjectContext,
)

__all__ = [
    # Multi-layer generators
    "LlmsTxtGenerator",
    "AgentsMdGenerator",
    "RepomapGenerator",
    "ProjectProfile",
    "ReadmeLlmGenerator",
    "NestedAgentsMdGenerator",
    "MermaidGenerator",
    "MermaidContext",
    # Main orchestrator
    "MultiLayerDocGenerator",
    "DocumentationGenerator",
    "AnalysisResult",
    "generate_documentation",
    # Legacy
    "AgentGuidelinesGenerator",
    "generate_agent_guidelines",
    "ProjectContext",
]
