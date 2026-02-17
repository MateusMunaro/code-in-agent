"""
LangGraph nodes for the code analysis agent.

The agent uses a reasoning loop with these nodes:
1. ReadStructureNode - Analyzes folder structure and identifies key files
2. PlanningNode - Plans what to investigate next
3. VerificationNode - Reads files and verifies hypotheses
4. EmbeddingsNode - Generates embeddings for code chunks
5. SemanticSearchNode - Semantic search for relevant code
6. ResponseNode - Generates final documentation
"""

import json
import re
import asyncio
import numpy as np
from typing import Literal, Optional
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState, ReasoningStep, PatternInfo, FileContent, CodeChunk, SemanticResult, detect_architecture_from_structure
from ..llm.provider import MultiModelChat
from ..services.git_service import GitService
from ..services.stub_builder import StubBuilder


class BaseNode:
    """Base class for LangGraph nodes."""

    # Pastas que devem ser ignoradas na análise para economizar tokens
    IGNORED_DIRS = {
        'node_modules', 'dist', 'build', 'venv', '__pycache__',
        '.git', '.vscode', '.idea', 'coverage', '.next', '.cache',
        'vendor', 'target', 'out', 'bin', 'obj', '.mypy_cache',
        '.pytest_cache', '.tox', 'eggs', '*.egg-info'
    }

    def __init__(self, chat: MultiModelChat, git_service: GitService):
        self.chat = chat
        self.git_service = git_service

    def log(self, message: str):
        """Log a message with the node name."""
        print(f"[{self.__class__.__name__}] {message}")

    def _parse_llm_json(self, text: str) -> dict:
        """
        Limpa o Markdown antes de fazer o parse do JSON.
        
        As LLMs (Gemini, Claude, etc) frequentemente respondem com blocos
        de código markdown (```json ... ```) que quebram o json.loads().
        
        Args:
            text: Texto raw da resposta da LLM
            
        Returns:
            Dicionário parseado do JSON
            
        Raises:
            json.JSONDecodeError: Se o JSON ainda for inválido após limpeza
        """
        try:
            # Remove blocos de código ```json ... ``` ou ``` ... ```
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\w*\s*', '', text)  # Remove qualquer ```linguagem
            text = re.sub(r'```', '', text)
            
            # Remove espaços em branco extras
            text = text.strip()
            
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.log(f"JSON Parse Error. Raw content (truncated): {text[:200]}...")
            raise e


class ReadStructureNode(BaseNode):
    """
    Node that reads and analyzes the repository structure.
    Identifies key files and initial architecture patterns.
    """

    async def __call__(self, state: AgentState) -> dict:
        self.log("Analyzing repository structure...")

        file_tree = state["file_tree"]
        dep_graph = state["dependency_graph"]
        
        self.log(f"\ud83d\udcca Input: {len(file_tree)} files, {len(dep_graph.get('nodes', []))} dep nodes, {len(dep_graph.get('edges', []))} dep edges")

        # Detect architecture from folder structure
        arch_candidates = detect_architecture_from_structure(file_tree)

        # Identify key files to read
        key_files = self._identify_key_files(file_tree, dep_graph)

        # Create summary for LLM
        structure_summary = self._create_structure_summary(file_tree, dep_graph)

        # Ask LLM to analyze structure
        messages = [
            SystemMessage(content="""You are an expert software architect analyzing a codebase.
Based on the repository structure provided, identify:
1. The likely architecture pattern (e.g., Clean Architecture, MVC, Hexagonal, Microservices)
2. Key entry points and core modules
3. Important files that should be read to understand the codebase

Respond in JSON format:
{
    "architecture_hypothesis": "string",
    "confidence": 0.0-1.0,
    "key_modules": ["list of important directories/modules"],
    "files_to_read": ["list of file paths to read next"],
    "initial_observations": "string"
}"""),
            HumanMessage(content=f"""Repository structure:
{structure_summary}

Dependency graph stats:
- Total files: {dep_graph.get('stats', {}).get('file_count', 0)}
- Total functions: {dep_graph.get('stats', {}).get('function_count', 0)}
- Total classes: {dep_graph.get('stats', {}).get('class_count', 0)}

Architecture candidates from folder analysis:
{json.dumps(arch_candidates[:5], indent=2)}

Key files identified:
{json.dumps(key_files[:10], indent=2)}"""),
        ]

        try:
            response = await self.chat.ainvoke(messages)
            result = self._parse_llm_json(response.content)
        except (json.JSONDecodeError, Exception) as e:
            self.log(f"LLM response parsing failed: {e}")
            result = {
                "architecture_hypothesis": arch_candidates[0][0] if arch_candidates else "Unknown",
                "confidence": 0.3,
                "key_modules": [],
                "files_to_read": key_files[:5],
                "initial_observations": "Structure analysis completed with fallback.",
            }

        # Create reasoning step
        step = ReasoningStep(
            iteration=state["iteration"],
            node="ReadStructureNode",
            action="Analyzed repository structure",
            observation=result.get("initial_observations", "Structure analyzed"),
            confidence_delta=result.get("confidence", 0.3),
        )

        return {
            "architecture_hypothesis": result.get("architecture_hypothesis"),
            "confidence": result.get("confidence", 0.3),
            "files_to_read": result.get("files_to_read", key_files[:5]),
            "reasoning_steps": [step],
            "iteration": state["iteration"] + 1,
        }

    def _identify_key_files(self, file_tree: list, dep_graph: dict) -> list[str]:
        """Identify the most important files to read."""
        key_files = []

        # Priority 1: Entry points (files not imported by others)
        entry_points = set()
        imported_files = set()
        
        for edge in dep_graph.get("edges", []):
            if edge["type"] == "imports":
                imported_files.add(edge["target"].replace("file:", ""))

        for node in dep_graph.get("nodes", []):
            if node["type"] == "file":
                path = node["path"]
                if path not in imported_files:
                    entry_points.add(path)

        # Priority 2: Config files
        config_patterns = [
            "package.json", "pyproject.toml", "setup.py",
            "tsconfig.json", "next.config", "vite.config",
            "docker-compose", "Dockerfile",
        ]

        # Priority 3: Main/index files
        main_patterns = [
            "main.py", "main.ts", "main.js",
            "index.ts", "index.js", "index.py",
            "app.ts", "app.js", "app.py",
            "server.ts", "server.js",
        ]

        for f in file_tree:
            path = f["path"]
            name = Path(path).name.lower()

            # Config files
            if any(p in name for p in config_patterns):
                key_files.append(path)
            # Main files
            elif any(p in name for p in main_patterns):
                key_files.append(path)
            # Entry points
            elif path in entry_points:
                key_files.append(path)

        return list(set(key_files))[:15]

    def _create_structure_summary(
        self, 
        file_tree: list, 
        dep_graph: dict, 
        max_files_per_dir: int = 5,
        max_total_lines: int = 100
    ) -> str:
        """
        Create a summary of the repository structure.
        
        Otimizado para evitar context overflow em projetos grandes.
        
        Args:
            file_tree: Lista de FileInfo dicts
            dep_graph: Grafo de dependências
            max_files_per_dir: Máximo de arquivos mostrados por diretório
            max_total_lines: Máximo de linhas no sumário total
            
        Returns:
            String com resumo da estrutura
        """
        # Group files by directory
        dirs = {}
        for f in file_tree:
            parts = Path(f["path"]).parts
            if len(parts) > 1:
                top_dir = parts[0]
                if top_dir not in dirs:
                    dirs[top_dir] = []
                dirs[top_dir].append(f["path"])

        summary_parts = []
        total_lines = 0
        
        for dir_name, files in sorted(dirs.items()):
            # Pula pastas irrelevantes ou ocultas para economizar tokens
            if dir_name.startswith('.') or dir_name.lower() in self.IGNORED_DIRS:
                continue
            
            # Verifica se já excedemos o limite de linhas
            if total_lines >= max_total_lines:
                summary_parts.append(f"\n... (e mais {len(dirs) - len([d for d in summary_parts if d.startswith('📁')])} diretórios truncados)")
                break
            
            summary_parts.append(f"📁 {dir_name}/ ({len(files)} files)")
            total_lines += 1
            
            # Mostra apenas os top X arquivos
            for f in files[:max_files_per_dir]:
                if total_lines >= max_total_lines:
                    break
                summary_parts.append(f"   - {f}")
                total_lines += 1
            
            if len(files) > max_files_per_dir:
                summary_parts.append(f"   ... ({len(files) - max_files_per_dir} more hidden)")
                total_lines += 1

        return "\n".join(summary_parts)


class PlanningNode(BaseNode):
    """
    Node that plans the next investigation steps.
    Decides which files to read based on current understanding.
    """

    async def __call__(self, state: AgentState) -> dict:
        self.log(f"Planning iteration {state['iteration']}...")

        # Get current understanding
        files_read = state["files_read"]
        files_to_read = state["files_to_read"]
        patterns = state["patterns_detected"]
        hypothesis = state["architecture_hypothesis"]
        confidence = state["confidence"]

        # If we've read enough or confidence is high, skip to response
        max_iterations = state.get("max_iterations", 5)
        if confidence >= 0.8 or state["iteration"] >= max_iterations:
            self.log(f"Sufficient confidence ({confidence:.2f}) or max iterations reached")
            return {"files_to_read": []}

        # Ask LLM what to investigate next
        messages = [
            SystemMessage(content="""You are an expert software architect investigating a codebase.
Based on what you've learned so far, decide what to investigate next.

Consider:
1. Files that would confirm or refute the architecture hypothesis
2. Core business logic files
3. Configuration and dependency injection setup
4. Test files that reveal intended behavior

Respond in JSON format:
{
    "next_files": ["list of file paths to read"],
    "investigation_goal": "what you want to learn",
    "hypothesis_update": "any updates to architecture hypothesis",
    "confidence_adjustment": -0.1 to 0.1
}"""),
            HumanMessage(content=f"""Current understanding:

Architecture Hypothesis: {hypothesis}
Confidence: {confidence:.2f}

Files already read ({len(files_read)}):
{json.dumps([f['path'] for f in files_read], indent=2)}

Patterns detected:
{json.dumps([{'name': p['name'], 'confidence': p['confidence']} for p in patterns], indent=2)}

Files suggested for reading:
{json.dumps(files_to_read[:10], indent=2)}

What should we investigate next?"""),
        ]

        try:
            response = await self.chat.ainvoke(messages)
            result = self._parse_llm_json(response.content)
        except (json.JSONDecodeError, Exception) as e:
            self.log(f"Planning failed: {e}")
            # Fallback: just use the next files in queue
            result = {
                "next_files": files_to_read[:3],
                "investigation_goal": "Continue reading key files",
                "hypothesis_update": None,
                "confidence_adjustment": 0,
            }

        # Update files to read
        next_files = result.get("next_files", [])
        
        # Filter out already-read files
        read_paths = {f["path"] for f in files_read}
        next_files = [f for f in next_files if f not in read_paths]

        step = ReasoningStep(
            iteration=state["iteration"],
            node="PlanningNode",
            action=f"Planned to read {len(next_files)} files",
            observation=result.get("investigation_goal", "Continuing investigation"),
            confidence_delta=result.get("confidence_adjustment", 0),
        )

        new_confidence = max(0, min(1, confidence + result.get("confidence_adjustment", 0)))

        return {
            "files_to_read": next_files,
            "confidence": new_confidence,
            "reasoning_steps": [step],
        }


class VerificationNode(BaseNode):
    """
    Node that reads files and verifies the architecture hypothesis.
    Uses Map-Reduce pattern to process files in parallel, avoiding context overflow.
    
    Fase MAP: Analisa cada arquivo individualmente em paralelo
    Fase REDUCE: Agrega os resultados de todas as análises
    """
    
    # Tamanho máximo de conteúdo por arquivo para análise individual
    MAX_CONTENT_PER_FILE = 3000
    
    async def __call__(self, state: AgentState) -> dict:
        self.log(f"Verifying hypothesis with file reading (Map-Reduce)...")

        files_to_read = state["files_to_read"][:5]  # Limit files per iteration
        repo_path = state["repo_path"]
        hypothesis = state["architecture_hypothesis"]
        file_tree = state.get("file_tree", [])

        self.log(f"\ud83d\udcc2 Files to read: {files_to_read}")
        self.log(f"\ud83d\udcca Already have {len(state.get('files_read', []))} files_read, confidence={state.get('confidence', 0):.2f}")

        if not files_to_read:
            self.log("No files to read")
            return {}

        # Build file_tree index for quick lookup
        tree_index = {f.get("path", ""): f for f in file_tree}
        stub_builder = StubBuilder()

        # Read files — prefer stubs from file_tree metadata, fallback to raw content
        files_content = []
        for file_path in files_to_read:
            file_meta = tree_index.get(file_path)
            
            # Try stub-based content first (semantic approach)
            if file_meta and (file_meta.get("function_details") or file_meta.get("class_details")):
                content = stub_builder.build_file_stub(file_meta)
                self.log(f"\ud83e\udde9 Stub generated for {file_path} ({len(content)} chars vs {file_meta.get('line_count', '?')} original lines)")
            else:
                # Fallback: read raw content (unsupported languages or no metadata)
                content = await self.git_service.get_file_content(repo_path, file_path)
                if content and len(content) > 5000:
                    content = content[:5000] + "\n... (truncated)"

            if content:
                files_content.append(FileContent(
                    path=file_path,
                    content=content,
                    language=file_meta.get("language", Path(file_path).suffix) if file_meta else Path(file_path).suffix,
                    summary=None,
                ))

        if not files_content:
            self.log("Could not read any files")
            return {"files_to_read": []}

        # ============================================
        # FASE MAP: Analisar cada arquivo em paralelo
        # ============================================
        self.log(f"MAP Phase: Analyzing {len(files_content)} files in parallel...")
        
        tasks = []
        for file_content in files_content:
            tasks.append(self._analyze_single_file(file_content, hypothesis))
        
        # Executa todas as análises ao mesmo tempo (muito mais rápido)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # ============================================
        # FASE REDUCE: Agregar os resultados
        # ============================================
        self.log(f"REDUCE Phase: Aggregating results...")
        
        aggregated_patterns = []
        all_observations = []
        all_improvements = []
        total_confidence_delta = 0.0
        successful_analyses = 0
        
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.log(f"Analysis of file {i} failed: {res}")
                continue
            
            if res:
                aggregated_patterns.extend(res.get("patterns_found", []))
                if res.get("observations"):
                    all_observations.append(res.get("observations"))
                all_improvements.extend(res.get("improvements", []))
                total_confidence_delta += res.get("confidence_delta", 0.1)
                successful_analyses += 1
        
        # Calcular delta médio de confiança
        avg_confidence_delta = total_confidence_delta / max(successful_analyses, 1)
        
        # Criar PatternInfo objects dos padrões agregados
        new_patterns = self._merge_patterns(state["patterns_detected"], aggregated_patterns)
        
        # Update confidence
        new_confidence = max(0, min(1, state["confidence"] + avg_confidence_delta))
        
        # Combinar observações
        combined_observations = " | ".join(all_observations[:3])  # Limitar para evitar overflow
        if len(all_observations) > 3:
            combined_observations += f" ... (+{len(all_observations) - 3} more observations)"

        step = ReasoningStep(
            iteration=state["iteration"],
            node="VerificationNode",
            action=f"Read and analyzed {len(files_content)} files (Map-Reduce with stubs)",
            observation=combined_observations[:200] if combined_observations else "Files analyzed",
            confidence_delta=avg_confidence_delta,
        )

        return {
            "files_read": files_content,
            "patterns_detected": new_patterns,
            "confidence": new_confidence,
            "improvements": list(set(all_improvements)),  # Remove duplicates
            "reasoning_steps": [step],
            "files_to_read": [],  # Clear the queue
        }
    
    async def _analyze_single_file(
        self, 
        file_content: FileContent, 
        hypothesis: str
    ) -> Optional[dict]:
        """
        Analisa UM arquivo isoladamente (Prompt Leve).
        
        Esta é a fase MAP do Map-Reduce. Cada arquivo é analisado
        individualmente para evitar context overflow.
        
        Args:
            file_content: Conteúdo do arquivo a ser analisado
            hypothesis: Hipótese de arquitetura atual
            
        Returns:
            Dict com patterns_found, observations, confidence_delta, improvements
            ou None se falhar
        """
        # Truncar conteúdo para o limite
        # Content is already a compact stub (or truncated raw), no need to re-truncate
        content = file_content["content"]
        
        prompt = f"""Analyze this single file for the architecture hypothesis: "{hypothesis}"
        
File: {file_content['path']}
Language: {file_content['language']}
```
{content}
```

Provide a brief analysis in JSON format:
{{
    "patterns_found": [{"name": "string", "description": "string", "evidence": ["{file_content['path']}"]}],
    "observations": "one-line summary of key observations",
    "confidence_delta": -0.1 to 0.2,
    "improvements": ["list of improvements if any"]
}}"""
        
        try:
            response = await self.chat.ainvoke([HumanMessage(content=prompt)])
            return self._parse_llm_json(response.content)
        except Exception as e:
            self.log(f"Error analyzing {file_content['path']}: {e}")
            return None
    
    def _merge_patterns(
        self, 
        existing_patterns: list[PatternInfo], 
        new_patterns_raw: list[dict]
    ) -> list[PatternInfo]:
        """
        Merge existing patterns with newly discovered patterns.
        
        Evita duplicatas e aumenta a confiança de padrões já detectados.
        
        Args:
            existing_patterns: Padrões já detectados anteriormente
            new_patterns_raw: Novos padrões em formato dict
            
        Returns:
            Lista consolidada de PatternInfo
        """
        # Mapear padrões existentes por nome
        pattern_map = {p["name"].lower(): p for p in existing_patterns}
        
        for raw_pattern in new_patterns_raw:
            name = raw_pattern.get("name", "").lower()
            
            if name in pattern_map:
                # Padrão já existe, aumentar confiança e adicionar evidência
                existing = pattern_map[name]
                new_evidence = raw_pattern.get("evidence", [])
                existing["evidence"] = list(set(existing["evidence"] + new_evidence))
                existing["confidence"] = min(existing["confidence"] + 0.1, 1.0)
            else:
                # Novo padrão encontrado
                new_pattern = PatternInfo(
                    name=raw_pattern.get("name", "Unknown"),
                    description=raw_pattern.get("description", ""),
                    evidence=raw_pattern.get("evidence", []),
                    confidence=0.6,  # Confiança inicial mais baixa para novos padrões
                )
                pattern_map[name] = new_pattern
        
        return list(pattern_map.values())


class ResponseNode(BaseNode):
    """
    Node that generates the final documentation proposal.
    
    Uses the multi-layer documentation architecture to generate:
    - Root Level: llms.txt, AGENTS.md, repomap.txt
    - Module Level: {module}/ReadMe.LLM, {module}/AGENTS.md
    - Semantic Layer: _codein/* (Phase 3)
    
    This creates AI-friendly documentation that helps coding assistants
    understand and work with the codebase effectively.
    """

    async def __call__(self, state: AgentState) -> dict:
        self.chat.set_task_context("Response")
        self.log("Generating structured documentation...")

        # Import the documentation generator
        from ..templates import generate_documentation

        # Compile all information from analysis
        files_read = state["files_read"]
        patterns = state["patterns_detected"]
        hypothesis = state["architecture_hypothesis"]
        reasoning = state["reasoning_steps"]
        improvements = state.get("improvements", [])
        dep_graph = state["dependency_graph"]
        file_tree = state["file_tree"]
        repo_path = state.get("repo_path", "")

        # === DETAILED LOGGING ===
        self.log(f"📊 Data pipeline status:")
        self.log(f"   file_tree: {len(file_tree)} files")
        self.log(f"   files_read: {len(files_read)} files")
        self.log(f"   patterns_detected: {len(patterns)} patterns")
        self.log(f"   dependency_graph nodes: {len(dep_graph.get('nodes', []))}")
        self.log(f"   dependency_graph edges: {len(dep_graph.get('edges', []))}")
        self.log(f"   hypothesis: {hypothesis}")
        self.log(f"   improvements: {len(improvements)}")

        # Log file_tree detail sample
        if file_tree:
            sample = file_tree[0]
            funcs = len(sample.get('function_details', []))
            classes = len(sample.get('class_details', []))
            self.log(f"   file_tree[0] sample: path={sample.get('path', '?')}, funcs={funcs}, classes={classes}")
            total_funcs = sum(len(f.get('function_details', [])) for f in file_tree)
            total_classes = sum(len(f.get('class_details', [])) for f in file_tree)
            self.log(f"   file_tree totals: {total_funcs} functions, {total_classes} classes across {len(file_tree)} files")

        # Detect main language and framework
        main_language, framework = self._detect_language_and_framework(files_read, file_tree)
        self.log(f"🔍 Detected language: {main_language}, framework: {framework}")
        
        # Extract project name from repo path
        project_name = Path(repo_path).name if repo_path else "Project"

        # Generate directory structure summary
        directory_structure = self._generate_directory_summary(file_tree)
        
        # Extract entry points and key modules
        entry_points = self._extract_entry_points(dep_graph)
        key_modules = self._extract_key_modules(file_tree, dep_graph)
        self.log(f"🔍 Entry points: {entry_points[:5]}")
        self.log(f"🔍 Key modules: {key_modules[:5]}")

        # Extract tech stack
        tech_stack = self._extract_tech_stack(files_read)
        self.log(f"🔍 Tech stack: {tech_stack}")

        # Extract config file contents (reads from filesystem, not files_read)
        config_contents = self._extract_config_contents(repo_path)
        self.log(f"📁 Config files found: {list(config_contents.keys())}")

        # Generate structured documentation using our templates
        self.log("📝 Using structured documentation templates...")
        
        # Get job_id from state for storage upload
        job_id = state.get("job_id", "")
        storage_path = None
        documentation_files = []
        
        # Initialize PR variables
        pr_url = None
        pr_number = None
        pr_branch = None
        
        try:
            # Generate the comprehensive documentation (multiple files)
            docs_dict = generate_documentation(
                project_name=project_name,
                architecture_pattern=hypothesis,
                confidence=state["confidence"],
                main_language=main_language,
                files_read=[{"path": f["path"], "content": f.get("content", ""), "summary": f.get("summary", "")} for f in files_read],
                patterns_detected=patterns,
                dependency_graph=dep_graph,
                directory_structure=directory_structure,
                framework=framework,
                tech_stack=tech_stack,
                improvements=improvements,
                entry_points=entry_points,
                key_modules=key_modules,
                file_tree=file_tree,
                code_chunks=state.get("code_chunks", []),
                config_files_content=config_contents,
                output_format="full",  # Generate full documentation structure
            )
            
            self.log(f"✅ Generated {len(docs_dict)} documentation files")
            
            # Upload to Supabase Storage if job_id is available
            if job_id and isinstance(docs_dict, dict):
                try:
                    from ..services.storage_service import get_storage_service
                    storage_service = get_storage_service()
                    storage_path, documentation_files = storage_service.upload_documentation(
                        job_id=job_id,
                        docs=docs_dict
                    )
                    self.log(f"Uploaded {len(documentation_files)} files to storage: {storage_path}")
                except Exception as storage_error:
                    self.log(f"Storage upload failed: {storage_error}, continuing with inline docs")
            
            # Create Pull Request if GitHub token is available
            github_token = state.get("github_token")
            self.log(f"GitHub token available: {bool(github_token)}")
            
            if github_token and isinstance(docs_dict, dict):
                try:
                    from ..services.github_service import GitHubService
                    github_service = GitHubService(github_token)
                    repo_url = state.get("repo_url", "")
                    
                    self.log(f"Creating Pull Request for {repo_url}...")
                    pr_result = await github_service.create_documentation_pr(
                        repo_url=repo_url,
                        documentation_files=docs_dict,
                        pr_title=f"📚 Add AI-generated documentation for {project_name}",
                    )
                    
                    if pr_result.success:
                        pr_url = pr_result.pr_url
                        pr_number = pr_result.pr_number
                        pr_branch = pr_result.branch_name
                        self.log(f"✅ PR created: {pr_url}")
                    else:
                        self.log(f"⚠️ PR creation failed: {pr_result.error}")
                except Exception as pr_error:
                    self.log(f"PR creation error: {pr_error}")
            
            # Create summary documentation for backward compatibility
            # (concatenate key files or use generate_summary_documentation)
            if isinstance(docs_dict, dict):
                documentation = self._create_summary_from_full_docs(docs_dict)
            else:
                documentation = docs_dict
            
        except Exception as e:
            self.log(f"Structured documentation failed: {e}, using enhanced fallback")
            documentation = self._generate_enhanced_fallback_docs(
                state, project_name, main_language, framework
            )

        # Optionally enhance with LLM for more context-specific content
        if state["confidence"] >= 0.7 and isinstance(documentation, str):
            documentation = await self._enhance_with_llm(state, documentation)

        step = ReasoningStep(
            iteration=state["iteration"],
            node="ResponseNode",
            action="Generated structured documentation",
            observation=f"Created {len(documentation_files) if documentation_files else 1} documentation files",
            confidence_delta=0,
        )

        return {
            "documentation": documentation,
            "documentation_files": documentation_files,
            "storage_path": storage_path,
            "architecture_type": hypothesis,
            "reasoning_steps": [step],
            # PR information
            "pr_url": pr_url,
            "pr_number": pr_number,
            "pr_branch": pr_branch,
        }

    def _detect_language_and_framework(
        self, 
        files_read: list, 
        file_tree: list
    ) -> tuple[str, Optional[str]]:
        """Detect the main language and framework from analyzed files."""
        
        # Count file extensions
        extensions = {}
        for f in file_tree:
            ext = Path(f["path"]).suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        # Determine main language
        lang_map = {
            ".py": "Python",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".js": "JavaScript",
            ".jsx": "JavaScript",
            ".java": "Java",
            ".kt": "Kotlin",
            ".go": "Go",
            ".rs": "Rust",
            ".rb": "Ruby",
            ".php": "PHP",
            ".cs": "C#",
            ".cpp": "C++",
            ".c": "C",
        }
        
        main_lang = "Unknown"
        max_count = 0
        
        for ext, count in extensions.items():
            if ext in lang_map and count > max_count:
                main_lang = lang_map[ext]
                max_count = count
        
        # Detect framework from file names and content
        framework = None
        file_names = [f["path"].lower() for f in file_tree]
        all_paths = " ".join(file_names)
        
        # Check for common frameworks
        if "next.config" in all_paths or "app/page.tsx" in all_paths:
            framework = "Next.js"
        elif "vite.config" in all_paths:
            framework = "Vite"
        elif "angular.json" in all_paths:
            framework = "Angular"
        elif "vue.config" in all_paths or "nuxt.config" in all_paths:
            framework = "Vue/Nuxt"
        elif "fastapi" in all_paths or any("fastapi" in (f.get("content", "") or "").lower() for f in files_read[:5]):
            framework = "FastAPI"
        elif "django" in all_paths:
            framework = "Django"
        elif "flask" in all_paths:
            framework = "Flask"
        elif "express" in all_paths:
            framework = "Express.js"
        elif "spring" in all_paths:
            framework = "Spring"
        elif "rails" in all_paths or "gemfile" in all_paths:
            framework = "Ruby on Rails"
        
        return main_lang, framework

    def _generate_directory_summary(self, file_tree: list, max_depth: int = 3) -> str:
        """Generate a clean directory tree summary."""
        dirs = {}
        
        for f in file_tree:
            parts = Path(f["path"]).parts
            if len(parts) > 0:
                top = parts[0]
                if top not in self.IGNORED_DIRS and not top.startswith('.'):
                    if top not in dirs:
                        dirs[top] = {"files": 0, "subdirs": set()}
                    dirs[top]["files"] += 1
                    if len(parts) > 1:
                        dirs[top]["subdirs"].add(parts[1])
        
        # Build tree string
        lines = []
        for dir_name in sorted(dirs.keys())[:15]:
            info = dirs[dir_name]
            subdirs = list(info["subdirs"])[:5]
            lines.append(f"{dir_name}/")
            for subdir in subdirs:
                lines.append(f"├── {subdir}/")
            if len(info["subdirs"]) > 5:
                lines.append(f"└── ... (+{len(info['subdirs']) - 5} more)")
            elif subdirs:
                lines.append(f"└── ({info['files']} files)")
        
        return "\n".join(lines) if lines else "Structure not available"

    def _extract_entry_points(self, dep_graph: dict) -> list[str]:
        """Extract entry points from dependency graph."""
        entry_points = []
        
        # Files that are not imported by others
        imported = set()
        all_files = set()
        
        for edge in dep_graph.get("edges", []):
            if edge.get("type") == "imports":
                imported.add(edge.get("target", "").replace("file:", ""))
        
        for node in dep_graph.get("nodes", []):
            if node.get("type") == "file":
                path = node.get("path", "")
                all_files.add(path)
                if path not in imported:
                    entry_points.append(path)
        
        # Also add common entry point patterns
        entry_patterns = ["main.py", "main.ts", "index.ts", "app.py", "server.py", "index.js"]
        for node in dep_graph.get("nodes", []):
            path = node.get("path", "")
            if any(p in path.lower() for p in entry_patterns):
                if path not in entry_points:
                    entry_points.append(path)
        
        return entry_points[:10]

    def _extract_key_modules(self, file_tree: list, dep_graph: dict) -> list[str]:
        """Extract key modules from file tree."""
        # Get top-level directories
        top_dirs = set()
        
        for f in file_tree:
            parts = Path(f["path"]).parts
            if len(parts) > 0:
                top = parts[0]
                if top not in self.IGNORED_DIRS and not top.startswith('.'):
                    top_dirs.add(top)
        
        # Filter to likely code directories
        code_dirs = [
            d for d in top_dirs 
            if d.lower() in ["src", "lib", "app", "api", "core", "services", 
                           "components", "modules", "utils", "helpers", "domain",
                           "application", "infrastructure", "adapters", "ports"]
            or d not in ["docs", "tests", "test", "scripts", "config", "assets"]
        ]
        
        return code_dirs[:10] if code_dirs else list(top_dirs)[:10]

    def _extract_tech_stack(self, files_read: list) -> dict:
        """Extract technology stack from analyzed files."""
        tech_stack = {}
        
        for f in files_read:
            path = f.get("path", "").lower()
            content = f.get("content", "") or ""
            
            # Check package.json
            if "package.json" in path:
                try:
                    pkg = json.loads(content)
                    deps = list(pkg.get("dependencies", {}).keys())[:5]
                    if deps:
                        tech_stack["Main Dependencies"] = ", ".join(deps)
                    dev_deps = list(pkg.get("devDependencies", {}).keys())[:5]
                    if dev_deps:
                        tech_stack["Dev Dependencies"] = ", ".join(dev_deps)
                    if pkg.get("scripts"):
                        tech_stack["Scripts"] = ", ".join(list(pkg["scripts"].keys())[:5])
                except:
                    pass
            
            # Check requirements.txt
            elif "requirements.txt" in path:
                deps = [line.split("==")[0].split(">=")[0].strip() 
                       for line in content.split("\n") if line.strip() and not line.startswith("#")][:5]
                if deps:
                    tech_stack["Python Packages"] = ", ".join(deps)
            
            # Check pyproject.toml
            elif "pyproject.toml" in path:
                if "poetry" in content.lower():
                    tech_stack["Package Manager"] = "Poetry"
                elif "pdm" in content.lower():
                    tech_stack["Package Manager"] = "PDM"
                elif "hatch" in content.lower():
                    tech_stack["Package Manager"] = "Hatch"
            
            # Check Cargo.toml
            elif "cargo.toml" in path:
                tech_stack["Build System"] = "Cargo"
                tech_stack["Language Runtime"] = "Rust"
            
            # Check go.mod
            elif "go.mod" in path:
                tech_stack["Build System"] = "Go Modules"
                tech_stack["Language Runtime"] = "Go"
            
            # Check Makefile / CMakeLists
            elif path.endswith("makefile") or path == "makefile":
                tech_stack["Build System"] = "Make"
            elif "cmakelists.txt" in path:
                tech_stack["Build System"] = "CMake"
        
        return tech_stack

    def _extract_config_contents(self, repo_path: str) -> dict:
        """Extract content of configuration files directly from the repository filesystem."""
        config_patterns = [
            "package.json", "requirements.txt", "pyproject.toml", "setup.py",
            "setup.cfg", "Cargo.toml", "go.mod", "go.sum",
            "Makefile", "makefile", "CMakeLists.txt", "meson.build",
            "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
            ".env.example", ".env.sample",
            "tsconfig.json", "vite.config.ts", "vite.config.js",
            "next.config.js", "next.config.mjs", "next.config.ts",
            "webpack.config.js", "rollup.config.js",
            "jest.config.js", "jest.config.ts", "vitest.config.ts",
            "pytest.ini", "tox.ini",
            ".eslintrc.json", ".eslintrc.js", ".prettierrc",
            "README.md", "README.txt", "README.rst",
        ]
        
        config_contents = {}
        
        if not repo_path:
            self.log("⚠️ No repo_path available for config extraction")
            return config_contents
        
        repo_root = Path(repo_path)
        if not repo_root.exists():
            self.log(f"⚠️ Repo path does not exist: {repo_path}")
            return config_contents
        
        for pattern in config_patterns:
            config_file = repo_root / pattern
            if config_file.is_file():
                try:
                    content = config_file.read_text(encoding='utf-8', errors='ignore')
                    if content:
                        config_contents[pattern] = content[:8000]  # Limit size
                        self.log(f"   📄 Found config: {pattern} ({len(content)} bytes)")
                except Exception as e:
                    self.log(f"   ⚠️ Failed to read {pattern}: {e}")
        
        # Also search one level deep for common config files
        for subdir in repo_root.iterdir():
            if subdir.is_dir() and subdir.name not in {
                'node_modules', '__pycache__', '.git', 'venv', 'dist', 'build',
                '.venv', 'target', '.next', 'coverage'
            }:
                for pattern in ['package.json', 'Cargo.toml', 'go.mod', 'Makefile', 'CMakeLists.txt']:
                    config_file = subdir / pattern
                    if config_file.is_file():
                        try:
                            content = config_file.read_text(encoding='utf-8', errors='ignore')
                            if content:
                                rel_path = f"{subdir.name}/{pattern}"
                                config_contents[rel_path] = content[:8000]
                                self.log(f"   📄 Found config: {rel_path} ({len(content)} bytes)")
                        except Exception:
                            pass
        
        self.log(f"📁 Total config files extracted: {len(config_contents)}")
        return config_contents

    async def _enhance_with_llm(self, state: AgentState, documentation: str) -> str:
        """Optionally enhance documentation with LLM for specific insights."""
        
        # Only enhance if we have good confidence and specific patterns
        patterns = state.get("patterns_detected", [])
        
        if len(patterns) < 2:
            return documentation
        
        # Ask LLM for specific architectural insights
        prompt = f"""Given this codebase analysis with architecture "{state['architecture_hypothesis']}" 
and patterns: {[p['name'] for p in patterns]},

Add a brief "## Architectural Insights" section (max 200 words) with:
1. Why this architecture choice makes sense
2. One key strength of the current approach
3. One potential improvement area

Keep it practical and actionable. Return only the section content in Markdown."""

        try:
            response = await self.chat.ainvoke([HumanMessage(content=prompt)])
            insights = response.content.strip()
            
            # Insert insights into documentation after architecture section
            if "## Detected Patterns" in documentation:
                documentation = documentation.replace(
                    "## Detected Patterns",
                    f"## Architectural Insights\n\n{insights}\n\n---\n\n## Detected Patterns"
                )
            else:
                documentation += f"\n\n---\n\n## Architectural Insights\n\n{insights}"
            
        except Exception as e:
            self.log(f"LLM enhancement failed (non-critical): {e}")
        
        return documentation

    def _create_summary_from_full_docs(self, docs_dict: dict[str, str]) -> str:
        """
        Create a summary documentation from the full docs dictionary.
        
        Combines key files into a single markdown for backward compatibility.
        Uses the new multi-layer documentation keys.
        
        Args:
            docs_dict: Dictionary of file paths to content
            
        Returns:
            Single markdown string with combined documentation
        """
        # Priority order — root governance files first
        priority_files = [
            "AGENTS.md",
            "llms.txt",
            "repomap.txt",
        ]
        
        summary_parts = []
        
        # Add priority files first
        for file_path in priority_files:
            if file_path in docs_dict:
                summary_parts.append(docs_dict[file_path])
        
        # Add module ReadMe.LLM files (limited to top 10)
        module_docs = sorted(k for k in docs_dict if k.endswith("/ReadMe.LLM"))
        for file_path in module_docs[:10]:
            summary_parts.append(docs_dict[file_path])
        
        # Add remaining files as appendix
        included = set(priority_files) | set(module_docs[:10])
        other_files = [k for k in docs_dict.keys() if k not in included]
        if other_files:
            summary_parts.append("\n---\n\n## Additional Documentation\n")
            summary_parts.append(f"\n*{len(other_files)} additional documentation files available in storage.*\n")
            summary_parts.append("\n### Available Files:\n")
            for f in sorted(other_files):
                summary_parts.append(f"- `{f}`\n")
        
        return "\n\n---\n\n".join(summary_parts) if summary_parts else "No documentation generated."


    def _generate_enhanced_fallback_docs(
        self, 
        state: AgentState, 
        project_name: str,
        main_language: str,
        framework: Optional[str]
    ) -> str:
        """Generate enhanced fallback documentation if main generation fails."""
        
        patterns_list = "\n".join(
            f"- **{p.get('name', 'Unknown')}**: {p.get('description', 'N/A')}"
            for p in state.get("patterns_detected", [])[:5]
        ) or "- No specific patterns detected"
        
        improvements_list = "\n".join(
            f"- {imp}" for imp in state.get("improvements", [])
        ) or "- No specific recommendations at this time"
        
        return f"""# 📚 Project Documentation - {project_name}

> Documentation automatically generated by Code Analysis Agent.

---

## Overview

**Project:** {project_name}  
**Architecture:** {state['architecture_hypothesis']}  
**Confidence:** {state['confidence']:.0%}  
**Language:** {main_language}  
{f"**Framework:** {framework}" if framework else ""}

---

## Architecture

This repository follows a **{state['architecture_hypothesis']}** architecture.

### Structure
- Total files analyzed: {len(state.get('file_tree', []))}
- Files read in detail: {len(state.get('files_read', []))}

---

## Detected Patterns

{patterns_list}

---

## Improvement Recommendations

{improvements_list}

---

## 🤖 Rules for AI Agents

### Before Modifying Code

1. Understand the **{state['architecture_hypothesis']}** architecture
2. Follow the detected patterns above
3. Check existing files before creating new ones

### Recommended Documentation Folder Structure

```
docs/
├── charts/           # Visual diagrams (Mermaid)
├── context/          # Reference documentation
├── usage/            # Practical guides
└── AGENT_RULES.md    # Rules for AI agents
```

---

*Generated by Code Analysis Agent*
"""


class EmbeddingsNode(BaseNode):
    """
    Node that generates embeddings for code chunks.
    
    Creates embeddings for functions, classes, and modules from the
    file_tree to enable semantic search in later nodes.
    """

    async def __call__(self, state: AgentState) -> dict:
        self.log("Generating embeddings for code chunks (using code stubs)...")

        file_tree = state.get("file_tree", [])
        repo_path = state.get("repo_path", "")
        
        if not file_tree:
            self.log("No files to embed")
            return {"embeddings_ready": False, "code_chunks": []}

        # Import embeddings service
        try:
            from ..llm.embeddings import EmbeddingService
            embedding_service = EmbeddingService()
        except Exception as e:
            self.log(f"Embedding service unavailable: {e}")
            return {"embeddings_ready": False, "code_chunks": []}

        # Use StubBuilder to generate code-native representations for embeddings
        stub_builder = StubBuilder()
        chunks = []
        
        for file_info in file_tree:
            file_path = file_info.get("path", "")
            language = file_info.get("language", "")
            
            # Process function details — embed as code stubs
            function_details = file_info.get("function_details", [])
            for func in function_details:
                if not isinstance(func, dict):
                    continue
                
                # Generate code-native stub instead of generic text
                content = stub_builder.build_function_stub(func, language)
                content += f"\n# File: {file_path}" if language == "python" else f"\n// File: {file_path}"
                
                chunks.append({
                    "content": content,
                    "file_path": file_path,
                    "language": language,
                    "chunk_type": "function",
                    "name": func.get("name", ""),
                    "start_line": func.get("start_line", 0),
                    "end_line": func.get("end_line", 0),
                    "embedding": None,
                })
            
            # Process class details — embed as code stubs
            class_details = file_info.get("class_details", [])
            for cls in class_details:
                if not isinstance(cls, dict):
                    continue
                
                # Generate code-native stub instead of generic text
                content = stub_builder.build_class_stub(cls, language)
                content += f"\n# File: {file_path}" if language == "python" else f"\n// File: {file_path}"
                
                chunks.append({
                    "content": content,
                    "file_path": file_path,
                    "language": language,
                    "chunk_type": "class",
                    "name": cls.get("name", ""),
                    "start_line": cls.get("start_line", 0),
                    "end_line": cls.get("end_line", 0),
                    "embedding": None,
                })

        if not chunks:
            self.log("No chunks to embed")
            return {"embeddings_ready": False, "code_chunks": []}

        self.log(f"Generating embeddings for {len(chunks)} code stubs...")

        try:
            # Generate embeddings in batches
            embedded_chunks = await embedding_service.embed_code_chunks(chunks)
            
            # Convert to CodeChunk format
            code_chunks = [
                CodeChunk(
                    content=c["content"],
                    file_path=c["file_path"],
                    language=c["language"],
                    chunk_type=c["chunk_type"],
                    name=c.get("name"),
                    start_line=c.get("start_line", 0),
                    end_line=c.get("end_line", 0),
                    embedding=c.get("embedding"),
                )
                for c in embedded_chunks
            ]
            
            self.log(f"Successfully generated {len(code_chunks)} embeddings from code stubs")
            
            step = ReasoningStep(
                iteration=state["iteration"],
                node="EmbeddingsNode",
                action=f"Generated embeddings for {len(code_chunks)} code stubs",
                observation=f"Embedded {len([c for c in code_chunks if c.get('chunk_type') == 'function'])} functions and {len([c for c in code_chunks if c.get('chunk_type') == 'class'])} classes as code stubs",
                confidence_delta=0.05,
            )
            
            return {
                "code_chunks": code_chunks,
                "embeddings_ready": True,
                "reasoning_steps": [step],
            }
            
        except Exception as e:
            self.log(f"Embedding generation failed: {e}")
            return {"embeddings_ready": False, "code_chunks": []}


class SemanticSearchNode(BaseNode):
    """
    Node that performs semantic search on code chunks.
    
    Given a query (from state or generated), finds the most relevant
    code chunks using cosine similarity on embeddings.
    """

    async def __call__(self, state: AgentState) -> dict:
        self.log("Performing semantic search...")

        code_chunks = state.get("code_chunks", [])
        embeddings_ready = state.get("embeddings_ready", False)
        hypothesis = state.get("architecture_hypothesis", "")
        
        if not embeddings_ready or not code_chunks:
            self.log("Embeddings not ready, skipping semantic search")
            return {"semantic_results": []}

        # Generate search query based on current hypothesis
        query = self._generate_search_query(state)
        
        if not query:
            self.log("No search query generated")
            return {"semantic_results": [], "semantic_query": None}

        self.log(f"Searching for: {query[:100]}...")

        try:
            from ..llm.embeddings import EmbeddingService
            embedding_service = EmbeddingService()
            
            # Get embedding for the query
            query_embedding = await embedding_service.embed_text(query)
            
            # Calculate cosine similarity with all chunks
            results = []
            for chunk in code_chunks:
                chunk_embedding = chunk.get("embedding")
                if chunk_embedding is None:
                    continue
                
                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                results.append({
                    "chunk": chunk,
                    "score": similarity,
                })
            
            # Sort by similarity and take top results
            results.sort(key=lambda x: x["score"], reverse=True)
            top_results = results[:10]  # Top 10 most relevant
            
            # Convert to SemanticResult format
            semantic_results = [
                SemanticResult(
                    chunk=r["chunk"],
                    score=r["score"],
                    relevance=self._score_to_relevance(r["score"]),
                )
                for r in top_results
            ]
            
            self.log(f"Found {len(semantic_results)} relevant code chunks")
            
            # Build scores string outside f-string to avoid backslash issue
            top_scores = [f"{r['score']:.2f}" for r in top_results[:3]]
            
            step = ReasoningStep(
                iteration=state["iteration"],
                node="SemanticSearchNode",
                action=f"Semantic search for: {query[:50]}...",
                observation=f"Found {len(semantic_results)} relevant chunks with scores {top_scores}",
                confidence_delta=0.05,
            )
            
            return {
                "semantic_query": query,
                "semantic_results": semantic_results,
                "reasoning_steps": [step],
            }
            
        except Exception as e:
            self.log(f"Semantic search failed: {e}")
            return {"semantic_results": [], "semantic_query": query}

    def _generate_search_query(self, state: AgentState) -> str:
        """Generate a search query based on current analysis state."""
        hypothesis = state.get("architecture_hypothesis", "")
        patterns = state.get("patterns_detected", [])
        
        # Build a query that will find relevant code
        query_parts = []
        
        if hypothesis:
            query_parts.append(f"Code implementing {hypothesis} architecture patterns")
        
        # Add pattern-specific queries
        pattern_names = [p.get("name", "") for p in patterns[:3] if isinstance(p, dict)]
        if pattern_names:
            query_parts.append(f"Functions and classes related to: {', '.join(pattern_names)}")
        
        # Add general architectural elements
        query_parts.append("Entry points, main handlers, core business logic")
        
        return " | ".join(query_parts) if query_parts else ""

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.0

    def _score_to_relevance(self, score: float) -> str:
        """Convert similarity score to relevance label."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"


