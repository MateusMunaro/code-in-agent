"""
Agent state definition for the LangGraph reasoning loop.
"""

from typing import TypedDict, Optional, Annotated
from operator import add


class FileContent(TypedDict):
    """Content of a file that was read by the agent."""
    path: str
    content: str
    language: str
    summary: Optional[str]


class ReasoningStep(TypedDict):
    """A single step in the agent's reasoning process."""
    iteration: int
    node: str
    action: str
    observation: str
    confidence_delta: float


class PatternInfo(TypedDict):
    """Information about a detected pattern."""
    name: str
    description: str
    evidence: list[str]
    confidence: float


class AgentState(TypedDict):
    """
    The state that flows through the LangGraph agent.
    
    This state is passed between nodes and updated as the agent
    progresses through its reasoning loop.
    """
    
    # Repository information
    repo_path: str
    repo_url: str
    
    # Parsed data from services
    file_tree: list[dict]  # List of FileInfo dicts from parser
    dependency_graph: dict  # Graph from graph_builder
    
    # Agent's working memory
    files_read: Annotated[list[FileContent], add]  # Files the agent has read
    files_to_read: list[str]  # Files the agent plans to read next
    
    # Pattern detection
    patterns_detected: Annotated[list[PatternInfo], add]
    architecture_hypothesis: Optional[str]  # e.g., "Clean Architecture", "MVC", "Hexagonal"
    
    # Confidence tracking
    confidence: float  # 0.0 to 1.0
    confidence_reasons: list[str]
    
    # Reasoning history
    reasoning_steps: Annotated[list[ReasoningStep], add]
    iteration: int
    max_iterations: int
    
    # Output
    documentation: Optional[str]
    architecture_type: Optional[str]
    improvements: list[str]
    
    # Error handling
    error: Optional[str]


def create_initial_state(
    repo_path: str,
    repo_url: str,
    file_tree: list[dict],
    dependency_graph: dict,
    max_iterations: int = 5
) -> AgentState:
    """
    Create the initial state for the agent.
    
    Args:
        repo_path: Path to the cloned repository
        repo_url: Original URL of the repository
        file_tree: Parsed file information
        dependency_graph: Dependency graph from graph_builder
        max_iterations: Maximum reasoning iterations
        
    Returns:
        Initial AgentState
    """
    return AgentState(
        repo_path=repo_path,
        repo_url=repo_url,
        file_tree=file_tree,
        dependency_graph=dependency_graph,
        files_read=[],
        files_to_read=[],
        patterns_detected=[],
        architecture_hypothesis=None,
        confidence=0.0,
        confidence_reasons=[],
        reasoning_steps=[],
        iteration=0,
        max_iterations=max_iterations,
        documentation=None,
        architecture_type=None,
        improvements=[],
        error=None,
    )


# Architecture patterns that the agent can detect
ARCHITECTURE_PATTERNS = {
    "clean_architecture": {
        "name": "Clean Architecture",
        "indicators": [
            "domain/", "entities/", "usecases/", "use_cases/",
            "infrastructure/", "adapters/", "ports/",
        ],
        "description": "Layers: Entities → Use Cases → Interface Adapters → Frameworks",
    },
    "hexagonal": {
        "name": "Hexagonal Architecture (Ports & Adapters)",
        "indicators": [
            "ports/", "adapters/", "domain/", "application/",
            "inbound/", "outbound/",
        ],
        "description": "Core domain with ports (interfaces) and adapters (implementations)",
    },
    "mvc": {
        "name": "MVC (Model-View-Controller)",
        "indicators": [
            "models/", "views/", "controllers/",
            "model/", "view/", "controller/",
        ],
        "description": "Traditional separation of Model, View, and Controller",
    },
    "mvvm": {
        "name": "MVVM (Model-View-ViewModel)",
        "indicators": [
            "viewmodels/", "viewmodel/", "models/", "views/",
        ],
        "description": "Model-View-ViewModel pattern, common in frontend apps",
    },
    "layered": {
        "name": "Layered Architecture",
        "indicators": [
            "presentation/", "business/", "data/",
            "api/", "service/", "repository/", "dao/",
        ],
        "description": "Traditional layered architecture with clear separation",
    },
    "microservices": {
        "name": "Microservices",
        "indicators": [
            "services/", "docker-compose", "kubernetes/",
            "k8s/", "gateway/", "api-gateway/",
        ],
        "description": "Distributed system with independent services",
    },
    "monolith": {
        "name": "Monolithic",
        "indicators": [
            "src/", "app/", "lib/",
        ],
        "description": "Single deployable unit with all functionality",
    },
    "modular": {
        "name": "Modular Monolith",
        "indicators": [
            "modules/", "features/", "packages/",
        ],
        "description": "Monolith organized into independent modules",
    },
    "ddd": {
        "name": "Domain-Driven Design",
        "indicators": [
            "domain/", "aggregates/", "entities/", "value_objects/",
            "repositories/", "services/", "events/",
        ],
        "description": "Focus on core domain logic with bounded contexts",
    },
    "cqrs": {
        "name": "CQRS (Command Query Responsibility Segregation)",
        "indicators": [
            "commands/", "queries/", "handlers/",
            "read_model/", "write_model/",
        ],
        "description": "Separate models for reading and writing data",
    },
    "serverless": {
        "name": "Serverless",
        "indicators": [
            "functions/", "lambda/", "handlers/",
            "serverless.yml", "netlify.toml", "vercel.json",
        ],
        "description": "Function-as-a-Service architecture",
    },
    "nextjs_app": {
        "name": "Next.js App Router",
        "indicators": [
            "app/", "layout.tsx", "page.tsx",
            "loading.tsx", "error.tsx", "next.config",
        ],
        "description": "Next.js application using the App Router pattern",
    },
    "express_api": {
        "name": "Express.js API",
        "indicators": [
            "routes/", "middlewares/", "controllers/",
            "express", "app.js", "server.js",
        ],
        "description": "REST API built with Express.js",
    },
}


def detect_architecture_from_structure(file_tree: list[dict]) -> list[tuple[str, float]]:
    """
    Detect possible architectures based on folder structure.
    
    Args:
        file_tree: List of FileInfo dicts
        
    Returns:
        List of (architecture_name, confidence) tuples sorted by confidence
    """
    paths = [f["path"].lower() for f in file_tree]
    path_str = " ".join(paths)

    scores = {}

    for arch_id, arch_info in ARCHITECTURE_PATTERNS.items():
        indicators = arch_info["indicators"]
        matches = sum(1 for ind in indicators if ind.lower() in path_str)
        
        if matches > 0:
            # Calculate confidence based on matches
            confidence = min(matches / len(indicators), 1.0)
            scores[arch_info["name"]] = confidence

    # Sort by confidence
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_scores
