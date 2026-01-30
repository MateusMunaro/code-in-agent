"""
Documentation templates for AI-friendly codebase documentation.

This module provides templates and structures for generating comprehensive
documentation that helps AI coding assistants understand and work with codebases.
"""

from .doc_structure import DocumentationStructure, DocTemplate, DocCategory
from .agent_guidelines import AgentGuidelinesGenerator, generate_agent_guidelines, ProjectContext
from .doc_generator import DocumentationGenerator, AnalysisResult, generate_documentation

__all__ = [
    # Structure
    "DocumentationStructure",
    "DocTemplate",
    "DocCategory",
    # Agent guidelines
    "AgentGuidelinesGenerator",
    "generate_agent_guidelines",
    "ProjectContext",
    # Documentation generator
    "DocumentationGenerator",
    "AnalysisResult",
    "generate_documentation",
]
