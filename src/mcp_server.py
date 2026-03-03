"""
MCP Server for Code-In.

Thin wrapper around the `code-in` CLI. Each tool simply spawns the CLI
as a subprocess with `--wait` and returns whatever the CLI outputs.

The CLI handles everything:
  scan → submit to backend → poll for result → print to stdout

Usage:
  python -m src.mcp_server                              # stdio (Claude Desktop, Cursor)
  python -m src.mcp_server --transport streamable-http   # HTTP

Requirements:
  pip install "mcp[server]"
  `code-in` CLI must be installed and in PATH (with --wait support)
"""

import os
import sys
import asyncio
import logging

from mcp.server.fastmcp import FastMCP

# =============================================
# Configuration
# =============================================
CLI_COMMAND = os.getenv("CODEIN_CLI", "code-in")
CLI_TIMEOUT = int(os.getenv("CODEIN_TIMEOUT", "300"))  # 5 minutes max

logger = logging.getLogger("code-in-mcp")

# =============================================
# MCP Server
# =============================================
mcp = FastMCP(
    "code-in",
    description=(
        "Code analysis and documentation agent. "
        "Scans local codebases, detects architecture patterns, "
        "builds dependency graphs, and generates documentation "
        "with improvement suggestions."
    ),
)


# =============================================
# Helper
# =============================================

async def _run_cli(args: list[str], cwd: str | None = None) -> str:
    """
    Run the code-in CLI and return its stdout.

    All CLI output intended for the user goes to stdout.
    Logs/debug info goes to stderr (ignored here).
    """
    cmd = [CLI_COMMAND] + args
    work_dir = cwd or os.getcwd()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=CLI_TIMEOUT,
        )
    except FileNotFoundError:
        return (
            f"❌ CLI '{CLI_COMMAND}' not found. "
            f"Make sure code-in is installed and in PATH, "
            f"or set CODEIN_CLI environment variable."
        )
    except asyncio.TimeoutError:
        proc.kill()
        return f"❌ CLI timed out after {CLI_TIMEOUT} seconds."

    output = stdout.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        error = stderr.decode("utf-8", errors="replace").strip()
        return f"❌ CLI error (exit {proc.returncode}): {error or output}"

    return output or "⚠️ CLI returned no output."


# =============================================
# Tools
# =============================================

@mcp.tool()
async def analyze_project(
    path: str = ".",
    project_name: str = "",
    include_content: bool = True,
) -> str:
    """
    Scan and analyze a local project directory with AI.

    Scans all code files, detects architecture patterns, builds
    dependency graphs, and generates comprehensive documentation
    with improvement suggestions.

    Args:
        path: Path to the project directory to analyze
        project_name: Optional name for the project
        include_content: Whether to include file contents in the analysis
    """
    args = ["scan", path, "--wait"]
    if project_name:
        args.extend(["--name", project_name])
    if not include_content:
        args.append("--no-content")

    return await _run_cli(args, cwd=os.path.abspath(path))


@mcp.tool()
async def analyze_changes(
    path: str = ".",
    project_name: str = "",
) -> str:
    """
    Analyze code changes since the last snapshot.

    Compares the current project state with a previously saved
    snapshot, then uses AI to analyze the impact of changes on
    architecture and code quality.

    Requires: a snapshot taken previously with save_snapshot.

    Args:
        path: Path to the project directory
        project_name: Optional name for the project
    """
    args = ["diff", path, "--wait"]
    if project_name:
        args.extend(["--name", project_name])

    return await _run_cli(args, cwd=os.path.abspath(path))


@mcp.tool()
async def get_project_context(
    path: str = ".",
    project_name: str = "",
    include_content: bool = True,
) -> str:
    """
    Get structured project context (file tree + statistics).

    Returns the project structure without AI analysis.
    Useful for understanding the codebase layout quickly.

    Args:
        path: Path to the project directory
        project_name: Optional name for the project
        include_content: Whether to include file contents
    """
    args = ["context", path]
    if project_name:
        args.extend(["--name", project_name])
    if not include_content:
        args.append("--no-content")

    return await _run_cli(args, cwd=os.path.abspath(path))


@mcp.tool()
async def save_snapshot(path: str = ".") -> str:
    """
    Save a snapshot of the current project state.

    Creates a baseline for later comparison with analyze_changes.

    Args:
        path: Path to the project directory
    """
    return await _run_cli(["snapshot", path], cwd=os.path.abspath(path))


@mcp.tool()
async def check_status(job_id: str) -> str:
    """
    Check the status of a previously submitted analysis job.

    Args:
        job_id: The UUID of the analysis job
    """
    return await _run_cli(["status", "-j", job_id])


# =============================================
# Entry point
# =============================================
def main():
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
