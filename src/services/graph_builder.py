"""
Graph builder service for constructing code dependency graphs.
"""

import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    """A node in the dependency graph."""
    id: str
    path: str
    type: str  # "file", "function", "class", "module"
    name: str
    language: str
    metadata: dict = field(default_factory=dict)


@dataclass 
class GraphEdge:
    """An edge in the dependency graph."""
    source: str
    target: str
    type: str  # "imports", "calls", "extends", "implements"
    metadata: dict = field(default_factory=dict)


class GraphBuilder:
    """
    Service for building dependency graphs from parsed code.
    
    Supports both in-memory storage (for small repos) and
    can export to formats suitable for pgvector or Neo4j.
    """

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def clear(self):
        """Clear the current graph."""
        self.nodes.clear()
        self.edges.clear()

    async def build_graph(self, repo_path: str, files_info: list) -> dict:
        """
        Build a dependency graph from parsed files.
        
        Args:
            repo_path: Path to the repository
            files_info: List of FileInfo dicts from parser
            
        Returns:
            Graph as a dictionary with nodes and edges
        """
        self.clear()

        def _build():
            # Create file nodes
            for file_info in files_info:
                node_id = f"file:{file_info['path']}"
                self.nodes[node_id] = GraphNode(
                    id=node_id,
                    path=file_info['path'],
                    type="file",
                    name=Path(file_info['path']).name,
                    language=file_info['language'],
                    metadata={
                        "size_bytes": file_info['size_bytes'],
                        "line_count": file_info['line_count'],
                        "functions": file_info['functions'],
                        "classes": file_info['classes'],
                    }
                )

                # Create function nodes
                for func_name in file_info['functions']:
                    func_id = f"function:{file_info['path']}:{func_name}"
                    self.nodes[func_id] = GraphNode(
                        id=func_id,
                        path=file_info['path'],
                        type="function",
                        name=func_name,
                        language=file_info['language'],
                    )
                    # Edge from file to function
                    self.edges.append(GraphEdge(
                        source=node_id,
                        target=func_id,
                        type="contains",
                    ))

                # Create class nodes
                for class_name in file_info['classes']:
                    class_id = f"class:{file_info['path']}:{class_name}"
                    self.nodes[class_id] = GraphNode(
                        id=class_id,
                        path=file_info['path'],
                        type="class",
                        name=class_name,
                        language=file_info['language'],
                    )
                    # Edge from file to class
                    self.edges.append(GraphEdge(
                        source=node_id,
                        target=class_id,
                        type="contains",
                    ))

            # Create import edges
            self._build_import_edges(files_info, repo_path)

            return self._to_dict()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _build)

    def _build_import_edges(self, files_info: list, repo_path: str):
        """Build edges for import relationships."""
        # Create a mapping of module names to files
        module_to_file = {}
        
        for file_info in files_info:
            path = Path(file_info['path'])
            
            # For Python files
            if file_info['language'] == "python":
                # Convert path to module name
                module_name = str(path.with_suffix("")).replace("/", ".").replace("\\", ".")
                module_to_file[module_name] = file_info['path']
                
                # Also add just the filename (for relative imports)
                module_to_file[path.stem] = file_info['path']

            # For JS/TS files
            elif file_info['language'] in ("javascript", "typescript"):
                # Store with and without extension
                module_to_file[str(path)] = file_info['path']
                module_to_file[str(path.with_suffix(""))] = file_info['path']
                
                # Store relative to common directories
                for base in ["src", "lib", "app"]:
                    if str(path).startswith(base):
                        rel = str(path).replace(f"{base}/", "")
                        module_to_file[rel] = file_info['path']
                        module_to_file[str(Path(rel).with_suffix(""))] = file_info['path']

        # Now build import edges
        for file_info in files_info:
            source_id = f"file:{file_info['path']}"
            
            for imp in file_info['imports']:
                # Normalize import path
                imp_normalized = imp.replace("./", "").replace("../", "")
                
                # Try to find matching file
                target_path = None
                
                # Direct match
                if imp_normalized in module_to_file:
                    target_path = module_to_file[imp_normalized]
                else:
                    # Try partial match
                    for module_name, file_path in module_to_file.items():
                        if module_name.endswith(imp_normalized) or imp_normalized.endswith(module_name):
                            target_path = file_path
                            break

                if target_path and target_path != file_info['path']:
                    target_id = f"file:{target_path}"
                    if target_id in self.nodes:
                        self.edges.append(GraphEdge(
                            source=source_id,
                            target=target_id,
                            type="imports",
                            metadata={"import_name": imp}
                        ))

    def _to_dict(self) -> dict:
        """Convert the graph to a dictionary format."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "path": node.path,
                    "type": node.type,
                    "name": node.name,
                    "language": node.language,
                    "metadata": node.metadata,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "metadata": edge.metadata,
                }
                for edge in self.edges
            ],
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "file_count": len([n for n in self.nodes.values() if n.type == "file"]),
                "function_count": len([n for n in self.nodes.values() if n.type == "function"]),
                "class_count": len([n for n in self.nodes.values() if n.type == "class"]),
            }
        }

    def find_entry_points(self) -> list[str]:
        """Find likely entry points (files with no incoming imports)."""
        incoming = set()
        for edge in self.edges:
            if edge.type == "imports":
                incoming.add(edge.target)

        entry_points = []
        for node_id, node in self.nodes.items():
            if node.type == "file" and node_id not in incoming:
                entry_points.append(node.path)

        return entry_points

    def find_most_imported(self, limit: int = 10) -> list[tuple[str, int]]:
        """Find the most imported files."""
        import_counts = {}
        
        for edge in self.edges:
            if edge.type == "imports":
                target = edge.target
                import_counts[target] = import_counts.get(target, 0) + 1

        sorted_imports = sorted(
            import_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_imports[:limit]

    def find_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in the graph."""
        # Build adjacency list for import edges only
        adj = {}
        for edge in self.edges:
            if edge.type == "imports":
                if edge.source not in adj:
                    adj[edge.source] = []
                adj[edge.source].append(edge.target)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]

            path.pop()
            rec_stack.remove(node)
            return None

        for node in adj:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    cycles.append(cycle)

        return cycles

    def get_file_dependencies(self, file_path: str) -> dict:
        """Get all dependencies for a specific file."""
        node_id = f"file:{file_path}"
        
        imports = []
        imported_by = []

        for edge in self.edges:
            if edge.type == "imports":
                if edge.source == node_id:
                    imports.append(edge.target.replace("file:", ""))
                elif edge.target == node_id:
                    imported_by.append(edge.source.replace("file:", ""))

        return {
            "file": file_path,
            "imports": imports,
            "imported_by": imported_by,
            "import_count": len(imports),
            "imported_by_count": len(imported_by),
        }

    def to_pgvector_format(self) -> list[dict]:
        """
        Convert graph to a format suitable for pgvector storage.
        Each node becomes a row with metadata for filtering.
        """
        rows = []
        
        for node in self.nodes.values():
            # Find related edges
            related_imports = [
                e.target for e in self.edges 
                if e.source == node.id and e.type == "imports"
            ]
            imported_by = [
                e.source for e in self.edges 
                if e.target == node.id and e.type == "imports"
            ]

            rows.append({
                "node_id": node.id,
                "path": node.path,
                "type": node.type,
                "name": node.name,
                "language": node.language,
                "imports": related_imports,
                "imported_by": imported_by,
                "metadata": node.metadata,
                # Placeholder for embedding - would be generated by LLM
                "embedding": None,
            })

        return rows

    def to_cypher_statements(self) -> list[str]:
        """
        Generate Cypher statements for Neo4j import.
        """
        statements = []

        # Create nodes
        for node in self.nodes.values():
            props = {
                "id": node.id,
                "path": node.path,
                "name": node.name,
                "language": node.language,
            }
            props_str = ", ".join([f'{k}: "{v}"' for k, v in props.items()])
            statements.append(
                f'CREATE (n:{node.type.capitalize()} {{{props_str}}})'
            )

        # Create edges
        for edge in self.edges:
            statements.append(
                f'MATCH (a {{id: "{edge.source}"}}), (b {{id: "{edge.target}"}}) '
                f'CREATE (a)-[:{edge.type.upper()}]->(b)'
            )

        return statements
