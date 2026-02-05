"""
LangGraph nodes for the code analysis agent.

The agent uses a reasoning loop with these nodes:
1. ReadStructureNode - Analyzes folder structure and identifies key files
2. PlanningNode - Plans what to investigate next
3. VerificationNode - Reads files and verifies hypotheses
4. ResponseNode - Generates final documentation
"""

import json
import re
import asyncio
from typing import Literal, Optional
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState, ReasoningStep, PatternInfo, FileContent, detect_architecture_from_structure
from ..llm.provider import MultiModelChat
from ..services.git_service import GitService


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

        if not files_to_read:
            self.log("No files to read")
            return {}

        # Read files
        files_content = []
        for file_path in files_to_read:
            content = await self.git_service.get_file_content(repo_path, file_path)
            if content:
                # Truncate very long files
                if len(content) > 5000:
                    content = content[:5000] + "\n... (truncated)"

                files_content.append(FileContent(
                    path=file_path,
                    content=content,
                    language=Path(file_path).suffix,
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
            action=f"Read and analyzed {len(files_content)} files (Map-Reduce)",
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
        content = file_content["content"][:self.MAX_CONTENT_PER_FILE]
        
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
    
    Uses structured documentation templates to generate:
    - Visual diagrams (Mermaid)
    - Architecture context
    - Usage guides
    - Agent rules
    
    This creates AI-friendly documentation that helps coding assistants
    understand and work with the codebase effectively.
    """

    async def __call__(self, state: AgentState) -> dict:
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

        # Detect main language and framework
        main_language, framework = self._detect_language_and_framework(files_read, file_tree)
        
        # Extract project name from repo path
        project_name = Path(repo_path).name if repo_path else "Project"

        # Generate directory structure summary
        directory_structure = self._generate_directory_summary(file_tree)
        
        # Extract entry points and key modules
        entry_points = self._extract_entry_points(dep_graph)
        key_modules = self._extract_key_modules(file_tree, dep_graph)

        # Generate structured documentation using our templates
        self.log("Using structured documentation templates...")
        
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
                files_read=[{"path": f["path"], "summary": f.get("summary", "")} for f in files_read],
                patterns_detected=patterns,
                dependency_graph=dep_graph,
                directory_structure=directory_structure,
                framework=framework,
                tech_stack=self._extract_tech_stack(files_read),
                improvements=improvements,
                entry_points=entry_points,
                key_modules=key_modules,
                output_format="full",  # Generate full documentation structure
            )
            
            self.log(f"Generated {len(docs_dict)} documentation files")
            
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
        
        return tech_stack

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
        
        Args:
            docs_dict: Dictionary of file paths to content
            
        Returns:
            Single markdown string with combined documentation
        """
        # Priority order for summary
        priority_files = [
            "docs/STRUCTURE.md",
            "docs/charts/01_ARCHITECTURE_OVERVIEW.md",
            "docs/charts/02_CLASS_DIAGRAM.md",
            "docs/usage/00_INDEX.md",
            "docs/AGENT_RULES.md",
        ]
        
        summary_parts = []
        
        # Add priority files first
        for file_path in priority_files:
            if file_path in docs_dict:
                content = docs_dict[file_path]
                summary_parts.append(content)
        
        # Add remaining files as appendix (limited)
        other_files = [k for k in docs_dict.keys() if k not in priority_files]
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


