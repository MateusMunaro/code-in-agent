"""
Agent Guidelines Generator — Backward-compatibility wrapper.

The actual implementation has been moved to ``root_level.AgentsMdGenerator``
as part of the multi-layer documentation architecture.

This module preserves the old public API so existing call-sites
(``generate_agent_guidelines()``, ``AgentGuidelinesGenerator``,
``ProjectContext``) continue to work without changes.
"""

from dataclasses import dataclass
from typing import Optional

from .root_level import AgentsMdGenerator, ProjectProfile


# ─── Legacy dataclass (maps 1:1 to root_level fields) ───────────

@dataclass
class ProjectContext:
    """Context information extracted from project analysis."""
    project_name: str
    architecture_pattern: str
    main_language: str
    framework: Optional[str] = None
    patterns_detected: list = None
    tech_stack: dict = None
    entry_points: list = None
    key_modules: list = None

    def __post_init__(self):
        if self.patterns_detected is None:
            self.patterns_detected = []
        if self.tech_stack is None:
            self.tech_stack = {}
        if self.entry_points is None:
            self.entry_points = []
        if self.key_modules is None:
            self.key_modules = []


# ─── Legacy class (thin delegate) ───────────────────────────────

class AgentGuidelinesGenerator:
    """
    Generates AI-friendly guidelines based on codebase analysis.
    Delegates to ``root_level.AgentsMdGenerator``.
    """

    def __init__(self, context: ProjectContext):
        self.context = context
        self._delegate = AgentsMdGenerator(
            project_name=context.project_name,
            architecture_pattern=context.architecture_pattern,
            main_language=context.main_language,
            framework=context.framework,
            patterns_detected=context.patterns_detected,
            tech_stack=context.tech_stack,
            entry_points=context.entry_points,
            key_modules=context.key_modules,
            profile=ProjectProfile(),  # default profile
            module_paths=[],
        )

    def generate_full_guidelines(self) -> str:
        """Generate complete AGENTS.md content."""
        return self._delegate.generate()


# ─── Legacy convenience function ────────────────────────────────

def generate_agent_guidelines(
    project_name: str,
    architecture_pattern: str,
    main_language: str,
    framework: Optional[str] = None,
    patterns_detected: list = None,
    tech_stack: dict = None,
    entry_points: list = None,
    key_modules: list = None,
) -> str:
    """
    Convenience function to generate agent guidelines.

    Returns:
        Complete agent guidelines as markdown string.
    """
    context = ProjectContext(
        project_name=project_name,
        architecture_pattern=architecture_pattern,
        main_language=main_language,
        framework=framework,
        patterns_detected=patterns_detected or [],
        tech_stack=tech_stack or {},
        entry_points=entry_points or [],
        key_modules=key_modules or [],
    )

    generator = AgentGuidelinesGenerator(context)
    return generator.generate_full_guidelines()
