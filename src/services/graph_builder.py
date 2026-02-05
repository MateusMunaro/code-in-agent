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
        Build a comprehensive dependency graph from parsed files.
        
        Creates nodes for files, functions, and classes, then builds
        edges for: imports, calls, extends (inheritance), implements.
        
        Args:
            repo_path: Path to the repository
            files_info: List of FileInfo dicts from parser (with function_details, class_details)
            
        Returns:
            Graph as a dictionary with nodes and edges
        """
        self.clear()

        def _build():
            # Build lookup maps for resolving references
            function_map = {}  # func_name -> list of func_ids
            class_map = {}     # class_name -> list of class_ids
            
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

                # Create function nodes with rich metadata
                function_details = file_info.get('function_details', [])
                for func_detail in function_details:
                    if isinstance(func_detail, dict):
                        func_name = func_detail.get('name', '')
                    else:
                        func_name = getattr(func_detail, 'name', '')
                    
                    if not func_name:
                        continue
                        
                    func_id = f"function:{file_info['path']}:{func_name}"
                    
                    # Store metadata from AST
                    metadata = {}
                    if isinstance(func_detail, dict):
                        metadata = {
                            "start_line": func_detail.get('start_line'),
                            "end_line": func_detail.get('end_line'),
                            "parameters": func_detail.get('parameters', []),
                            "is_async": func_detail.get('is_async', False),
                            "is_method": func_detail.get('is_method', False),
                            "parent_class": func_detail.get('parent_class'),
                            "docstring": func_detail.get('docstring'),
                        }
                    
                    self.nodes[func_id] = GraphNode(
                        id=func_id,
                        path=file_info['path'],
                        type="function",
                        name=func_name,
                        language=file_info['language'],
                        metadata=metadata,
                    )
                    
                    # Track for call resolution
                    if func_name not in function_map:
                        function_map[func_name] = []
                    function_map[func_name].append(func_id)
                    
                    # Edge from file to function
                    self.edges.append(GraphEdge(
                        source=node_id,
                        target=func_id,
                        type="contains",
                    ))
                
                # Fallback for files without function_details (legacy)
                if not function_details:
                    for func_name in file_info.get('functions', []):
                        func_id = f"function:{file_info['path']}:{func_name}"
                        self.nodes[func_id] = GraphNode(
                            id=func_id,
                            path=file_info['path'],
                            type="function",
                            name=func_name,
                            language=file_info['language'],
                        )
                        if func_name not in function_map:
                            function_map[func_name] = []
                        function_map[func_name].append(func_id)
                        self.edges.append(GraphEdge(
                            source=node_id,
                            target=func_id,
                            type="contains",
                        ))

                # Create class nodes with rich metadata
                class_details = file_info.get('class_details', [])
                for class_detail in class_details:
                    if isinstance(class_detail, dict):
                        class_name = class_detail.get('name', '')
                    else:
                        class_name = getattr(class_detail, 'name', '')
                    
                    if not class_name:
                        continue
                    
                    class_id = f"class:{file_info['path']}:{class_name}"
                    
                    # Store metadata from AST
                    metadata = {}
                    if isinstance(class_detail, dict):
                        metadata = {
                            "start_line": class_detail.get('start_line'),
                            "end_line": class_detail.get('end_line'),
                            "bases": class_detail.get('bases', []),
                            "methods": class_detail.get('methods', []),
                            "implements": class_detail.get('implements', []),
                            "docstring": class_detail.get('docstring'),
                        }
                    
                    self.nodes[class_id] = GraphNode(
                        id=class_id,
                        path=file_info['path'],
                        type="class",
                        name=class_name,
                        language=file_info['language'],
                        metadata=metadata,
                    )
                    
                    # Track for inheritance resolution
                    if class_name not in class_map:
                        class_map[class_name] = []
                    class_map[class_name].append(class_id)
                    
                    # Edge from file to class
                    self.edges.append(GraphEdge(
                        source=node_id,
                        target=class_id,
                        type="contains",
                    ))
                
                # Fallback for files without class_details (legacy)
                if not class_details:
                    for class_name in file_info.get('classes', []):
                        class_id = f"class:{file_info['path']}:{class_name}"
                        self.nodes[class_id] = GraphNode(
                            id=class_id,
                            path=file_info['path'],
                            type="class",
                            name=class_name,
                            language=file_info['language'],
                        )
                        if class_name not in class_map:
                            class_map[class_name] = []
                        class_map[class_name].append(class_id)
                        self.edges.append(GraphEdge(
                            source=node_id,
                            target=class_id,
                            type="contains",
                        ))

            # Create import edges
            self._build_import_edges(files_info, repo_path)
            
            # Create CALLS edges from function_details
            self._build_call_edges(files_info, function_map)
            
            # Create EXTENDS (inheritance) edges from class_details
            self._build_inheritance_edges(files_info, class_map)
            
            # Create IMPLEMENTS edges from class_details (for TS/Java interfaces)
            self._build_implements_edges(files_info, class_map)

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

    def _build_call_edges(self, files_info: list, function_map: dict):
        """
        Build CALLS edges between functions based on function_details.
        
        For each function, look at its `calls` list and resolve to target functions.
        """
        for file_info in files_info:
            function_details = file_info.get('function_details', [])
            
            for func_detail in function_details:
                if isinstance(func_detail, dict):
                    func_name = func_detail.get('name', '')
                    calls = func_detail.get('calls', [])
                else:
                    func_name = getattr(func_detail, 'name', '')
                    calls = getattr(func_detail, 'calls', [])
                
                if not func_name or not calls:
                    continue
                
                source_id = f"function:{file_info['path']}:{func_name}"
                
                for call_name in calls:
                    # Resolve call_name to function node(s)
                    # Simple name matching (could be improved with scope analysis)
                    base_call_name = call_name.split('.')[-1] if '.' in call_name else call_name
                    
                    if base_call_name in function_map:
                        # Prefer function in same file, otherwise take first match
                        target_ids = function_map[base_call_name]
                        
                        target_id = None
                        for tid in target_ids:
                            if file_info['path'] in tid:
                                target_id = tid
                                break
                        if not target_id and target_ids:
                            target_id = target_ids[0]
                        
                        if target_id and target_id != source_id:
                            self.edges.append(GraphEdge(
                                source=source_id,
                                target=target_id,
                                type="calls",
                                metadata={"call_expression": call_name}
                            ))

    def _build_inheritance_edges(self, files_info: list, class_map: dict):
        """
        Build EXTENDS edges between classes based on class_details.bases.
        
        For each class, look at its `bases` (parent classes) and resolve.
        """
        for file_info in files_info:
            class_details = file_info.get('class_details', [])
            
            for class_detail in class_details:
                if isinstance(class_detail, dict):
                    class_name = class_detail.get('name', '')
                    bases = class_detail.get('bases', [])
                else:
                    class_name = getattr(class_detail, 'name', '')
                    bases = getattr(class_detail, 'bases', [])
                
                if not class_name or not bases:
                    continue
                
                source_id = f"class:{file_info['path']}:{class_name}"
                
                for base_name in bases:
                    if base_name in class_map:
                        target_ids = class_map[base_name]
                        
                        # Prefer class in same file
                        target_id = None
                        for tid in target_ids:
                            if file_info['path'] in tid:
                                target_id = tid
                                break
                        if not target_id and target_ids:
                            target_id = target_ids[0]
                        
                        if target_id and target_id != source_id:
                            self.edges.append(GraphEdge(
                                source=source_id,
                                target=target_id,
                                type="extends",
                                metadata={"base_class": base_name}
                            ))

    def _build_implements_edges(self, files_info: list, class_map: dict):
        """
        Build IMPLEMENTS edges between classes and interfaces (for TS/Java).
        
        For each class, look at its `implements` list and resolve.
        """
        for file_info in files_info:
            class_details = file_info.get('class_details', [])
            
            for class_detail in class_details:
                if isinstance(class_detail, dict):
                    class_name = class_detail.get('name', '')
                    implements = class_detail.get('implements', [])
                else:
                    class_name = getattr(class_detail, 'name', '')
                    implements = getattr(class_detail, 'implements', [])
                
                if not class_name or not implements:
                    continue
                
                source_id = f"class:{file_info['path']}:{class_name}"
                
                for interface_name in implements:
                    if interface_name in class_map:
                        target_ids = class_map[interface_name]
                        
                        target_id = None
                        for tid in target_ids:
                            if file_info['path'] in tid:
                                target_id = tid
                                break
                        if not target_id and target_ids:
                            target_id = target_ids[0]
                        
                        if target_id and target_id != source_id:
                            self.edges.append(GraphEdge(
                                source=source_id,
                                target=target_id,
                                type="implements",
                                metadata={"interface": interface_name}
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
