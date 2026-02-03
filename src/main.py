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

from .config import settings
from .graph.graph import create_agent_graph
from .graph.state import create_initial_state
from .services.git_service import GitService
from .services.parser_service import ParserService
from .services.graph_builder import GraphBuilder


# Redis channels
CHANNELS = {
    "JOB_QUEUE": "code-indexer:jobs",
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
        """Update job status in Supabase."""
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
        }).execute()

    async def process_job(self, job_data: dict):
        """Process a single code indexing job."""
        job_id = job_data["job_id"]
        repo_url = job_data["repo_url"]
        selected_model = job_data.get("selected_model", "gpt-4o-mini")

        print(f"\n{'='*60}")
        print(f"📦 Processing job: {job_id}")
        print(f"📂 Repository: {repo_url}")
        print(f"🤖 Model: {selected_model}")
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

            # Step 3: Build dependency graph
            await self.publish_status(job_id, "processing", "Building dependency graph...", 50)
            dep_graph = await self.graph_builder.build_graph(repo_path, file_tree)
            print(f"✅ Built dependency graph with {len(dep_graph.get('nodes', []))} nodes")

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
            )

            final_state = await agent.ainvoke(initial_state)
            print(f"✅ Agent completed with confidence: {final_state.get('confidence', 0):.2f}")

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

    async def listen_for_jobs(self):
        """Listen to Redis list for new jobs."""
        print("👂 Listening for jobs...")

        while self.running:
            try:
                # BRPOP blocks until a message is available (or timeout)
                result = await self.redis_client.brpop(
                    CHANNELS["JOB_QUEUE"],
                    timeout=5
                )

                if result:
                    _, message = result
                    job_data = json.loads(message)
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
            await self.redis_client.close()

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
