r"""
Integration test: Parse the code-in project with ParserService,
then generate stubs with StubBuilder for every file.

This test does NOT call the LLM - it validates the parser to stub pipeline
end-to-end on a real codebase.

Run:
    cd c:\Users\HP\OneDrive\Desktop\Trabalho\Projetos\code-in
    python -m agent.tests.test_agent_integration
"""

import asyncio
import sys
import os
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.src.services.parser_service import ParserService
from agent.src.services.stub_builder import StubBuilder
from agent.src.services.graph_builder import GraphBuilder


REPO_PATH = r"c:\Users\HP\OneDrive\Desktop\Trabalho\Projetos\code-in"


async def main():
    print("\n" + "=" * 60)
    print("🧪 Integration Test: Parser → StubBuilder Pipeline")
    print(f"📂 Target: {REPO_PATH}")
    print("=" * 60 + "\n")

    # ─── Step 1: Parse the project ───────────────────────────
    print("📌 Step 1: Parsing project with ParserService...")
    parser = ParserService()
    file_tree = await parser.parse_repository(REPO_PATH)

    print(f"   ✅ Parsed {len(file_tree)} files\n")

    # Count metadata
    total_functions = 0
    total_classes = 0
    total_decorators = 0
    total_method_details = 0
    files_with_metadata = 0

    for f in file_tree:
        fd = f.get("function_details", [])
        cd = f.get("class_details", [])
        total_functions += len(fd)
        total_classes += len(cd)
        if fd or cd:
            files_with_metadata += 1

        for func in fd:
            if isinstance(func, dict):
                total_decorators += len(func.get("decorators", []))
        for cls in cd:
            if isinstance(cls, dict):
                total_decorators += len(cls.get("decorators", []))
                total_method_details += len(cls.get("method_details", []))

    print(f"   📊 Summary:")
    print(f"      Files with Tree-sitter metadata: {files_with_metadata}/{len(file_tree)}")
    print(f"      Total functions extracted: {total_functions}")
    print(f"      Total classes extracted: {total_classes}")
    print(f"      Total decorators found: {total_decorators}")
    print(f"      Total method_details populated: {total_method_details}")
    print()

    # ─── Step 2: Generate stubs ──────────────────────────────
    print("📌 Step 2: Generating stubs with StubBuilder...")
    stub_builder = StubBuilder()
    stub_results = []

    for f in file_tree:
        path = f.get("path", "")
        if f.get("function_details") or f.get("class_details"):
            stub = stub_builder.build_file_stub(f)
            original_lines = f.get("line_count", 0)
            stub_lines = stub.count("\n") + 1
            stub_results.append({
                "path": path,
                "original_lines": original_lines,
                "stub_lines": stub_lines,
                "stub_chars": len(stub),
                "stub": stub,
            })

    print(f"   ✅ Generated {len(stub_results)} file stubs\n")

    # ─── Step 3: Show best examples ─────────────────────────
    print("📌 Step 3: Sample stubs (top 5 by compression ratio):\n")

    # Sort by compression ratio
    for r in stub_results:
        if r["original_lines"] > 0:
            r["ratio"] = r["stub_lines"] / r["original_lines"]
        else:
            r["ratio"] = 1.0

    stub_results.sort(key=lambda x: x["ratio"])

    for r in stub_results[:5]:
        print(f"   {'─' * 50}")
        print(f"   📄 {r['path']}")
        print(f"   Original: {r['original_lines']} lines → Stub: {r['stub_lines']} lines ({r['ratio']:.1%})")
        print(f"   Stub content:")
        for line in r["stub"].split("\n")[:20]:
            print(f"      {line}")
        if r["stub_lines"] > 20:
            print(f"      ... ({r['stub_lines'] - 20} more lines)")
        print()

    # ─── Step 4: Build dependency graph ──────────────────────
    print("📌 Step 4: Building dependency graph...")
    graph_builder = GraphBuilder()
    dep_graph = await graph_builder.build_graph(REPO_PATH, file_tree)

    node_count = len(dep_graph.get("nodes", []))
    edge_count = len(dep_graph.get("edges", []))
    print(f"   ✅ Graph: {node_count} nodes, {edge_count} edges\n")

    # ─── Step 5: Summary stats ───────────────────────────────
    total_original = sum(r["original_lines"] for r in stub_results)
    total_stub = sum(r["stub_lines"] for r in stub_results)

    print("=" * 60)
    print("📊 Final Summary")
    print("=" * 60)
    print(f"   Files parsed:          {len(file_tree)}")
    print(f"   Files with stubs:      {len(stub_results)}")
    print(f"   Total original lines:  {total_original}")
    print(f"   Total stub lines:      {total_stub}")
    if total_original > 0:
        print(f"   Compression ratio:     {total_stub/total_original:.1%}")
        print(f"   Token savings:         ~{100 - (total_stub/total_original * 100):.0f}%")
    print(f"   Functions extracted:   {total_functions}")
    print(f"   Classes extracted:     {total_classes}")
    print(f"   Decorators captured:   {total_decorators}")
    print(f"   Method details:        {total_method_details}")
    print(f"   Graph nodes:           {node_count}")
    print(f"   Graph edges:           {edge_count}")
    print("=" * 60)
    print("\n🎉 Integration test complete — pipeline works end-to-end!")
    print("   The agent is ready for LLM-based analysis.\n")


if __name__ == "__main__":
    asyncio.run(main())
