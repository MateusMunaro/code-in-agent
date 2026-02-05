"""
LangGraph graph orchestration for the code analysis agent.

Creates the reasoning loop with embeddings and semantic search:
ReadStructure → Embeddings → Planning → Verification → SemanticSearch → (loop back or) Response
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .nodes import ReadStructureNode, PlanningNode, VerificationNode, ResponseNode, EmbeddingsNode, SemanticSearchNode
from ..llm.provider import MultiModelChat, get_chat_model
from ..services.git_service import GitService
from ..config import settings


def should_continue(state: AgentState) -> Literal["planning", "response"]:
    """
    Determine whether to continue reasoning or generate response.
    
    Decision criteria:
    - If confidence >= threshold: go to response
    - If max iterations reached: go to response
    - If error occurred: go to response
    - Otherwise: continue to planning
    """
    confidence = state.get("confidence", 0)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    error = state.get("error")
    
    threshold = settings.confidence_threshold
    
    if error:
        return "response"
    
    if confidence >= threshold:
        print(f"✅ Confidence {confidence:.2f} >= threshold {threshold:.2f}, generating response")
        return "response"
    
    if iteration >= max_iterations:
        print(f"⚠️ Max iterations ({max_iterations}) reached, generating response")
        return "response"
    
    print(f"🔄 Confidence {confidence:.2f} < threshold {threshold:.2f}, continuing analysis")
    return "planning"


def should_verify(state: AgentState) -> Literal["verification", "response"]:
    """
    Determine whether to verify (read files) or skip to response.
    """
    files_to_read = state.get("files_to_read", [])
    
    if not files_to_read:
        # No files to read, check if we should end
        confidence = state.get("confidence", 0)
        if confidence >= settings.confidence_threshold:
            return "response"
        # Try to continue anyway
        return "verification"
    
    return "verification"


def create_agent_graph(model_id: str = None) -> StateGraph:
    """
    Create the LangGraph agent for code analysis.
    
    Args:
        model_id: The LLM model to use (e.g., "gpt-4o", "claude-3-5-sonnet")
        
    Returns:
        Compiled StateGraph ready for invocation
    """
    # Initialize components
    chat = MultiModelChat(default_model=model_id)
    git_service = GitService(settings.repos_base_path)
    
    # Create nodes
    read_structure = ReadStructureNode(chat, git_service)
    embeddings = EmbeddingsNode(chat, git_service)
    planning = PlanningNode(chat, git_service)
    verification = VerificationNode(chat, git_service)
    semantic_search = SemanticSearchNode(chat, git_service)
    response = ResponseNode(chat, git_service)
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("read_structure", read_structure)
    workflow.add_node("embeddings", embeddings)
    workflow.add_node("planning", planning)
    workflow.add_node("verification", verification)
    workflow.add_node("semantic_search", semantic_search)
    workflow.add_node("response", response)
    
    # Set entry point
    workflow.set_entry_point("read_structure")
    
    # Add edges
    # After reading structure, generate embeddings
    workflow.add_edge("read_structure", "embeddings")
    
    # After embeddings, decide if we need more info or can respond
    workflow.add_conditional_edges(
        "embeddings",
        should_continue,
        {
            "planning": "planning",
            "response": "response",
        }
    )
    
    # After planning, go to verification or response
    workflow.add_conditional_edges(
        "planning",
        should_verify,
        {
            "verification": "verification",
            "response": "response",
        }
    )
    
    # After verification, do semantic search
    workflow.add_edge("verification", "semantic_search")
    
    # After semantic search, go back to planning or to response
    workflow.add_conditional_edges(
        "semantic_search",
        should_continue,
        {
            "planning": "planning",
            "response": "response",
        }
    )
    
    # Response is the end
    workflow.add_edge("response", END)
    
    # Compile and return
    return workflow.compile()


async def run_analysis(
    repo_path: str,
    repo_url: str,
    file_tree: list[dict],
    dependency_graph: dict,
    model_id: str = None,
    max_iterations: int = 5,
) -> AgentState:
    """
    Run the full code analysis pipeline.
    
    Args:
        repo_path: Path to the cloned repository
        repo_url: Original repository URL
        file_tree: Parsed file information from ParserService
        dependency_graph: Dependency graph from GraphBuilder
        model_id: LLM model to use
        max_iterations: Maximum reasoning iterations
        
    Returns:
        Final AgentState with documentation and analysis results
    """
    # Create initial state
    initial_state = create_initial_state(
        repo_path=repo_path,
        repo_url=repo_url,
        file_tree=file_tree,
        dependency_graph=dependency_graph,
        max_iterations=max_iterations,
    )
    
    # Create and run the graph
    graph = create_agent_graph(model_id)
    
    print("\n" + "="*60)
    print("🤖 Starting Code Analysis Agent")
    print(f"📂 Repository: {repo_url}")
    print(f"🧠 Model: {model_id or 'default'}")
    print(f"🔄 Max iterations: {max_iterations}")
    print("="*60 + "\n")
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state)
    
    print("\n" + "="*60)
    print("✅ Analysis Complete")
    print(f"📊 Final confidence: {final_state.get('confidence', 0):.2%}")
    print(f"🏗️ Architecture: {final_state.get('architecture_type', 'Unknown')}")
    print(f"📝 Documentation length: {len(final_state.get('documentation', ''))} chars")
    print("="*60 + "\n")
    
    return final_state


# For testing/debugging: visualize the graph
def get_graph_visualization() -> str:
    """Get a Mermaid diagram of the agent graph."""
    return """
```mermaid
graph TD
    A[Start] --> B[ReadStructureNode]
    B --> C[EmbeddingsNode]
    C --> D{Confidence >= 80%?}
    D -->|No| E[PlanningNode]
    D -->|Yes| I[ResponseNode]
    E --> F{Files to read?}
    F -->|Yes| G[VerificationNode]
    F -->|No| I
    G --> H[SemanticSearchNode]
    H --> D
    I --> J[End]
    
    style B fill:#e1f5fe
    style C fill:#e0f7fa
    style E fill:#fff3e0
    style G fill:#e8f5e9
    style H fill:#f3e5f5
    style I fill:#fce4ec
```
"""
