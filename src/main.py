"""
Main entry point for the Code Indexer Agent.
Listens to Redis for job messages and processes them.
"""

import asyncio
import json
import signal
import sys
from datetime import datetime

import redis.asyncio as redis
from supabase import create_client, Client

from .config import settings, MCP_RESULT_PREFIX, MCP_RESULT_TTL
from .graph.graph import create_agent_graph
from .graph.state import create_initial_state, create_mcp_analysis_state
from .services.git_service import GitService
from .services.parser_service import ParserService
from .services.graph_builder import GraphBuilder
from .services.conflict_analyzer import ConflictAnalyzer



# Redis channels
CHANNELS = {
    "JOB_QUEUE": "code-indexer:jobs",
    "MCP_JOBS": "code-indexer:mcp-jobs",
    "CONFLICT_JOBS": "code-indexer:conflict-jobs",
    "JOB_STATUS": "code-indexer:status",
    "JOB_COMPLETE": "code-indexer:complete",
}


class CodeIndexerWorker:
    """Worker that processes code indexing jobs."""

    def __init__(self):
        self.redis_client: redis.Redis | None = None
        self.supabase: Client | None = None
        self.git_service = GitService(settings.repos_base_path)
        self.parser_service = ParserService()
        self.graph_builder = GraphBuilder()
        self.running = True

    async def initialize(self):
        """Initialize connections to Redis and Supabase."""
        print("🚀 Initializing Code Indexer Worker...")

        # Connect to Redis
        self.redis_client = redis.from_url(settings.redis_url)
        await self.redis_client.ping()
        print("✅ Connected to Redis")

        # Connect to Supabase
        if settings.supabase_url and settings.supabase_service_key:
            self.supabase = create_client(
                settings.supabase_url,
                settings.supabase_service_key
            )
            print("✅ Connected to Supabase")
        else:
            print("⚠️ Supabase not configured, running in dry-run mode")

    async def publish_status(self, job_id: str, status: str, message: str = "", progress: int = 0):
        """Publish job status update to Redis."""
        if not self.redis_client:
            return

        status_msg = {
            "job_id": job_id,
            "status": status,
            "message": message,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.redis_client.publish(
            CHANNELS["JOB_STATUS"],
            json.dumps(status_msg)
        )

    async def publish_complete(self, job_id: str, success: bool, result: dict = None, error: str = None):
        """Publish job completion to Redis."""
        if not self.redis_client:
            return

        complete_msg = {
            "job_id": job_id,
            "success": success,
            "result": result,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.redis_client.publish(
            CHANNELS["JOB_COMPLETE"],
            json.dumps(complete_msg)
        )

    async def update_job_status(self, job_id: str, status: str, result: dict = None, error: str = None):
        """Update job status in Supabase.
        
        NOTE: user_id is NEVER included in update_data — the agent uses supabaseAdmin
        (service role) and must never overwrite user ownership set by the backend.
        """
        if not self.supabase:
            return

        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if result:
            update_data["result"] = result
        if error:
            update_data["error_message"] = error

        # Defensive: ensure user_id is never accidentally overwritten
        update_data.pop("user_id", None)

        self.supabase.table("jobs").update(update_data).eq("id", job_id).execute()

    async def save_analysis_result(self, job_id: str, analysis: dict):
        """Save analysis result to Supabase."""
        if not self.supabase:
            print(f"📝 Analysis result (dry-run): {json.dumps(analysis, indent=2)[:500]}...")
            return

        self.supabase.table("analysis_results").insert({
            "job_id": job_id,
            "documentation": analysis.get("documentation", ""),
            "documentation_files": analysis.get("documentation_files", []),
            "storage_path": analysis.get("storage_path"),
            "patterns": analysis.get("patterns", []),
            "architecture_type": analysis.get("architecture_type", "unknown"),
            "confidence_score": analysis.get("confidence_score", 0.0),
            "agent_reasoning": analysis.get("reasoning_steps", []),
            "dependencies_graph": analysis.get("dependencies_graph", {}),
            "suggested_improvements": analysis.get("suggested_improvements", []),
            # PR information
            "pr_url": analysis.get("pr_url"),
            "pr_number": analysis.get("pr_number"),
            "pr_branch": analysis.get("pr_branch"),
            "pr_status": analysis.get("pr_status", "none"),
        }).execute()

    async def process_job(self, job_data: dict):
        """Process a single code indexing job."""
        job_id = job_data["job_id"]
        repo_url = job_data["repo_url"]
        selected_model = job_data.get("selected_model", "gemini-2.5-flash")
        github_token = job_data.get("github_token")  # Token for PR creation
        # user_id is preserved from the job record — agent never overwrites it
        user_id = job_data.get("user_id")

        print(f"\n{'='*60}")
        print(f"📦 Processing job: {job_id}")
        print(f"📂 Repository: {repo_url}")
        print(f"🤖 Model: {selected_model}")
        print(f"👤 User: {user_id or '(no user_id in message)'}")
        print(f"🔑 GitHub Token: {'✓ provided' if github_token else '✗ not provided'}")
        print(f"{'='*60}\n")

        try:
            # Step 1: Clone repository
            await self.publish_status(job_id, "processing", "Cloning repository...", 10)
            repo_path = await self.git_service.clone_repository(repo_url, job_id)
            print(f"✅ Cloned to: {repo_path}")

            # Step 2: Parse code structure
            await self.publish_status(job_id, "processing", "Parsing code structure...", 30)
            file_tree = await self.parser_service.parse_repository(repo_path)
            print(f"✅ Parsed {len(file_tree)} files")
            
            # Log AST data quality
            total_funcs = sum(len(f.get('function_details', [])) for f in file_tree)
            total_classes = sum(len(f.get('class_details', [])) for f in file_tree)
            total_imports = sum(len(f.get('imports', [])) for f in file_tree)
            languages = set(f.get('language', '?') for f in file_tree)
            print(f"📊 AST data: {total_funcs} functions, {total_classes} classes, {total_imports} imports")
            print(f"📊 Languages: {languages}")
            if file_tree:
                sample = file_tree[0]
                print(f"📊 Sample file: {sample.get('path', '?')} -> funcs={len(sample.get('function_details', []))}, classes={len(sample.get('class_details', []))}")

            # Step 3: Build dependency graph
            await self.publish_status(job_id, "processing", "Building dependency graph...", 50)
            dep_graph = await self.graph_builder.build_graph(repo_path, file_tree)
            print(f"✅ Built dependency graph with {len(dep_graph.get('nodes', []))} nodes, {len(dep_graph.get('edges', []))} edges")

            # Step 4: Run LangGraph agent
            await self.publish_status(job_id, "processing", "Analyzing with AI agent...", 70)
            agent = create_agent_graph(selected_model)
            
            # Usar create_initial_state para garantir que todas as chaves obrigatórias
            # do AgentState estejam presentes (incluindo max_iterations)
            initial_state = create_initial_state(
                repo_path=repo_path,
                repo_url=repo_url,
                file_tree=file_tree,
                dependency_graph=dep_graph,
                max_iterations=5,  # Limite de iterações para evitar loops infinitos
                job_id=job_id,  # Pass job_id for storage uploads
                github_token=github_token,  # Pass token for PR creation
            )

            final_state = await agent.ainvoke(initial_state)
            print(f"✅ Agent completed with confidence: {final_state.get('confidence', 0):.2f}")

            # Check if PR was created
            pr_url = final_state.get("pr_url")
            if pr_url:
                print(f"✅ Pull Request created: {pr_url}")

            # Step 5: Generate documentation
            await self.publish_status(job_id, "processing", "Generating documentation...", 90)
            
            analysis_result = {
                "documentation": final_state.get("documentation", ""),
                "documentation_files": final_state.get("documentation_files", []),
                "storage_path": final_state.get("storage_path"),
                "patterns": final_state.get("patterns_detected", []),
                "architecture_type": final_state.get("architecture_type", "unknown"),
                "confidence_score": final_state.get("confidence", 0.0),
                "reasoning_steps": final_state.get("reasoning_steps", []),
                "dependencies_graph": dep_graph,
                "suggested_improvements": final_state.get("improvements", []),
                # PR information
                "pr_url": final_state.get("pr_url"),
                "pr_number": final_state.get("pr_number"),
                "pr_branch": final_state.get("pr_branch"),
                "pr_status": "created" if final_state.get("pr_url") else "none",
            }

            # Save to Supabase
            await self.save_analysis_result(job_id, analysis_result)
            await self.update_job_status(job_id, "completed", result={"success": True})
            await self.publish_complete(job_id, True, result=analysis_result)

            print(f"✅ Job {job_id} completed successfully!")

            # Cleanup
            await self.git_service.cleanup(repo_path)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            
            await self.update_job_status(job_id, "failed", error=error_msg)
            await self.publish_complete(job_id, False, error=error_msg)

    # ─────────────────────────────────────────────
    # MCP Analysis Handler (Dry-Run / Read-Only)
    # ─────────────────────────────────────────────

    async def process_mcp_job(self, mcp_data: dict):
        """
        Process an MCP analysis job (read-only, no clone/branch/PR).
        
        Flow: Payload → Build Context → LangGraph Analysis → Redis Result
        """
        job_id = mcp_data["job_id"]
        payload = mcp_data.get("payload", {})
        payload_type = mcp_data.get("payload_type", "scan_result")
        selected_model = payload.get("metadata", {}).get("model", "gemini-2.5-flash")
        # user_id is preserved from the job record — agent never overwrites it
        user_id = mcp_data.get("user_id")

        print(f"\n{'='*60}")
        print(f"🔍 Processing MCP Analysis job: {job_id}")
        print(f"📋 Payload type: {payload_type}")
        print(f"🤖 Model: {selected_model}")
        print(f"👤 User: {user_id or '(no user_id in message)'}")
        print(f"{'='*60}\n")

        try:
            # Step 1: Build context from CLI payload (no clone!)
            await self.publish_status(job_id, "processing", "Building context from local payload...", 10)
            
            initial_state = create_mcp_analysis_state(
                payload=payload,
                job_id=job_id,
                max_iterations=3,  # Fewer iterations for analysis-only
            )
            
            file_count = len(initial_state.get("file_tree", []))
            print(f"✅ Built context: {file_count} files from payload")

            # Step 2: Build dependency graph from payload file_tree
            await self.publish_status(job_id, "processing", "Building dependency graph...", 30)
            dep_graph = await self.graph_builder.build_graph("", initial_state["file_tree"])
            initial_state["dependency_graph"] = dep_graph
            print(f"✅ Dependency graph: {len(dep_graph.get('nodes', []))} nodes")

            # Step 3: Run LangGraph agent (analysis only, no PR)
            await self.publish_status(job_id, "processing", "Analyzing with AI agent...", 50)
            agent = create_agent_graph(selected_model)
            final_state = await agent.ainvoke(initial_state)
            print(f"✅ Agent completed with confidence: {final_state.get('confidence', 0):.2f}")

            # Step 4: Build result string
            await self.publish_status(job_id, "processing", "Building analysis result...", 90)
            
            analysis_parts = []
            
            doc = final_state.get("documentation", "")
            if doc:
                analysis_parts.append(doc)
            
            arch = final_state.get("architecture_type")
            if arch:
                analysis_parts.append(f"\n## Architecture: {arch}")
            
            patterns = final_state.get("patterns_detected", [])
            if patterns:
                pattern_text = "\n## Detected Patterns\n"
                for p in patterns:
                    name = p.get("name", "Unknown") if isinstance(p, dict) else str(p)
                    desc = p.get("description", "") if isinstance(p, dict) else ""
                    pattern_text += f"- **{name}**: {desc}\n"
                analysis_parts.append(pattern_text)
            
            improvements = final_state.get("improvements", [])
            if improvements:
                imp_text = "\n## Suggested Improvements\n"
                for imp in improvements:
                    imp_text += f"- {imp}\n"
                analysis_parts.append(imp_text)
            
            result_text = "\n".join(analysis_parts) if analysis_parts else "Analysis completed but no documentation was generated."

            analysis_result = {
                "documentation": final_state.get("documentation", result_text),
                "documentation_files": final_state.get("documentation_files", []),
                "storage_path": final_state.get("storage_path"),
                "patterns": final_state.get("patterns_detected", []),
                "architecture_type": final_state.get("architecture_type", "unknown"),
                "confidence_score": final_state.get("confidence", 0.0),
                "reasoning_steps": final_state.get("reasoning_steps", []),
                "dependencies_graph": dep_graph,
                "suggested_improvements": final_state.get("improvements", []),
                "pr_url": final_state.get("pr_url"),
                "pr_number": final_state.get("pr_number"),
                "pr_branch": final_state.get("pr_branch"),
                "pr_status": "created" if final_state.get("pr_url") else "none",
                "payload_type": payload_type,
                "summary": result_text,
            }

            await self.save_analysis_result(job_id, analysis_result)

            # Step 5: Write result to Redis with TTL
            result_value = json.dumps({
                "status": "COMPLETED",
                "result": analysis_result,
            })
            await self.redis_client.set(
                f"{MCP_RESULT_PREFIX}{job_id}",
                result_value,
                ex=MCP_RESULT_TTL,
            )
            print(f"✅ Result written to Redis key: {MCP_RESULT_PREFIX}{job_id} (TTL: {MCP_RESULT_TTL}s)")

            # Also update Supabase if available
            await self.update_job_status(job_id, "completed", result=analysis_result)
            await self.publish_complete(job_id, True, result=analysis_result)

            print(f"✅ MCP Analysis job {job_id} completed successfully!")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ MCP Analysis job {job_id} failed: {error_msg}")

            # Write failure to Redis with TTL
            try:
                fail_value = json.dumps({
                    "status": "FAILED",
                    "error": error_msg,
                })
                await self.redis_client.set(
                    f"{MCP_RESULT_PREFIX}{job_id}",
                    fail_value,
                    ex=MCP_RESULT_TTL,
                )
            except Exception as redis_err:
                print(f"❌ Failed to write error to Redis: {redis_err}")

            await self.update_job_status(job_id, "failed", error=error_msg)
            await self.publish_complete(job_id, False, error=error_msg)

    # ─────────────────────────────────────────────
    # Conflict Analysis Handler
    # ─────────────────────────────────────────────

    async def process_conflict_job(self, job_data: dict):
        """
        Process a multi-branch conflict analysis job.

        Flow:
        1. Clone repository (full, no shallow)
        2. Validate requested branches exist
        3. Compute diffs between all branch pairs
        4. Run semantic analysis + LLM conflict detection
        5. Save results to Supabase + Redis
        """
        job_id = job_data["job_id"]
        repo_url = job_data["repo_url"]
        branches = job_data.get("branches", [])
        selected_model = job_data.get("selected_model", "gemini-2.5-flash")
        user_id = job_data.get("user_id")

        print(f"\n{'='*60}")
        print(f"🔀 Processing Conflict Analysis job: {job_id}")
        print(f"📂 Repository: {repo_url}")
        print(f"🌿 Branches: {branches}")
        print(f"🤖 Model: {selected_model}")
        print(f"👤 User: {user_id or '(no user_id)'}")
        print(f"{'='*60}\n")

        try:
            # Step 1: Full clone (need all branches)
            await self.publish_status(job_id, "processing", "Cloning repository (full)...", 10)
            repo_path = await self.git_service.clone_repository_full(repo_url, job_id)
            print(f"✅ Full clone completed: {repo_path}")

            # Step 2: Run conflict analysis
            await self.publish_status(job_id, "processing", "Analyzing branches for conflicts...", 30)

            from .llm.provider import MultiModelChat
            chat = MultiModelChat(default_model=selected_model)
            analyzer = ConflictAnalyzer(chat, self.git_service)

            state = await analyzer.analyze(
                repo_url=repo_url,
                repo_path=repo_path,
                branches=branches,
                job_id=job_id,
                model_id=selected_model,
            )

            if state.get("error"):
                raise RuntimeError(state["error"])

            print(f"✅ Conflict analysis completed:")
            print(f"   🔍 Risks found: {len(state['conflict_risks'])}")
            print(f"   📊 Overlapping files: {len(state['overlapping_files'])}")
            print(f"   📐 Merge order: {state['merge_order_suggestion']}")

            # Step 3: Build result
            await self.publish_status(job_id, "processing", "Building analysis report...", 80)
            summary = analyzer.build_result_summary(state)

            analysis_result = {
                "documentation": summary,
                "documentation_files": [],
                "storage_path": None,
                "patterns": [],
                "architecture_type": "conflict_analysis",
                "confidence_score": state.get("confidence", 0.0),
                "reasoning_steps": [],
                "dependencies_graph": {},
                "suggested_improvements": state.get("general_recommendations", []),
                # Conflict-specific data
                "conflict_analysis": {
                    "branches": state["branches"],
                    "conflict_risks": state["conflict_risks"],
                    "overlapping_files": state["overlapping_files"],
                    "branch_files": state["branch_files"],
                    "semantic_context": state["semantic_context"],
                    "merge_order_suggestion": state["merge_order_suggestion"],
                    "general_recommendations": state["general_recommendations"],
                },
                "pr_url": None,
                "pr_number": None,
                "pr_branch": None,
                "pr_status": "none",
            }

            # Save to Supabase
            await self.save_analysis_result(job_id, analysis_result)
            await self.update_job_status(job_id, "completed", result={"success": True})
            await self.publish_complete(job_id, True, result=analysis_result)

            print(f"✅ Conflict Analysis job {job_id} completed successfully!")

            # Cleanup
            await self.git_service.cleanup(repo_path)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Conflict Analysis job {job_id} failed: {error_msg}")

            await self.update_job_status(job_id, "failed", error=error_msg)
            await self.publish_complete(job_id, False, error=error_msg)

    async def listen_for_jobs(self):
        """Listen to Redis lists for new jobs (standard + MCP + conflict)."""
        print("👂 Listening for jobs on queues:")
        print(f"   📦 Standard:  {CHANNELS['JOB_QUEUE']}")
        print(f"   🔍 MCP:       {CHANNELS['MCP_JOBS']}")
        print(f"   🔀 Conflict:  {CHANNELS['CONFLICT_JOBS']}")

        while self.running:
            try:
                # BRPOP blocks until a message is available on any queue
                # Priority: MCP > Conflict > Standard
                result = await self.redis_client.brpop(
                    [
                        CHANNELS["MCP_JOBS"],
                        CHANNELS["CONFLICT_JOBS"],
                        CHANNELS["JOB_QUEUE"],
                    ],
                    timeout=5
                )

                if result:
                    queue_name, message = result
                    queue_str = queue_name.decode() if isinstance(queue_name, bytes) else queue_name
                    job_data = json.loads(message)

                    if queue_str == CHANNELS["MCP_JOBS"]:
                        await self.process_mcp_job(job_data)
                    elif queue_str == CHANNELS["CONFLICT_JOBS"]:
                        await self.process_conflict_job(job_data)
                    else:
                        await self.process_job(job_data)

            except json.JSONDecodeError as e:
                print(f"❌ Invalid job message: {e}")
            except Exception as e:
                print(f"❌ Error processing job: {e}")
                await asyncio.sleep(1)

    async def shutdown(self):
        """Graceful shutdown."""
        print("\n🛑 Shutting down worker...")
        self.running = False

        if self.redis_client:
            await self.redis_client.aclose()

        print("👋 Worker stopped")

    async def run(self):
        """Main run loop."""
        await self.initialize()

        # Handle shutdown signals
        loop = asyncio.get_event_loop()

        def signal_handler():
            asyncio.create_task(self.shutdown())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        await self.listen_for_jobs()


async def main():
    """Entry point."""
    worker = CodeIndexerWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
