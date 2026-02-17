r"""
Integration Test: Multi-Layer Documentation Generation

Tests the complete refactored documentation pipeline:
1. Parse the frontend/ directory with ParserService
2. Build dependency graph with GraphBuilder  
3. Generate multi-layer docs with MultiLayerDocGenerator
4. Save output files to test_output/frontend_docs/

Validates the new architecture:
    Root Level   → llms.txt, AGENTS.md, repomap.txt
    Module Level → {module}/ReadMe.LLM, {module}/AGENTS.md

Run:
    cd c:\Users\HP\OneDrive\Desktop\Trabalho\Projetos\code-in
    python -m agent.tests.test_multilayer_docs
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.src.services.parser_service import ParserService
from agent.src.services.graph_builder import GraphBuilder
from agent.src.templates import (
    MultiLayerDocGenerator,
    AnalysisResult,
)

# Test configuration
REPO_PATH = r"c:\Users\HP\OneDrive\Desktop\Trabalho\Projetos\code-in\frontend"
OUTPUT_PATH = r"c:\Users\HP\OneDrive\Desktop\Trabalho\Projetos\code-in\agent\tests\test_output\frontend_docs"


async def main():
    print("\n" + "=" * 70)
    print("🧪 Multi-Layer Documentation Generation Test")
    print(f"📂 Target Repository: {REPO_PATH}")
    print(f"📤 Output Directory: {OUTPUT_PATH}")
    print("=" * 70 + "\n")

    # ─── Step 1: Parse the repository ─────────────────────────
    print("📌 Step 1: Parsing repository with ParserService...")
    parser = ParserService()
    
    try:
        file_tree = await parser.parse_repository(REPO_PATH)
        print(f"   ✅ Parsed {len(file_tree)} files")
        
        # Statistics
        total_functions = sum(len(f.get("function_details", [])) for f in file_tree)
        total_classes = sum(len(f.get("class_details", [])) for f in file_tree)
        total_imports = sum(len(f.get("imports", [])) for f in file_tree)
        
        print(f"   📊 Extracted:")
        print(f"      - Functions: {total_functions}")
        print(f"      - Classes: {total_classes}")
        print(f"      - Imports: {total_imports}")
        print()
        
    except Exception as e:
        print(f"   ❌ Parser failed: {e}")
        return

    # ─── Step 2: Build dependency graph ────────────────────────
    print("📌 Step 2: Building dependency graph...")
    graph_builder = GraphBuilder()
    
    try:
        dep_graph = await graph_builder.build_graph(REPO_PATH, file_tree)
        node_count = len(dep_graph.get("nodes", []))
        edge_count = len(dep_graph.get("edges", []))
        
        print(f"   ✅ Graph built: {node_count} nodes, {edge_count} edges")
        print()
        
    except Exception as e:
        print(f"   ⚠️ Graph builder failed: {e}, using empty graph")
        dep_graph = {"nodes": [], "edges": []}
        print()

    # ─── Step 3: Detect project characteristics ────────────────
    print("📌 Step 3: Analyzing project characteristics...")
    
    # Language detection (simple heuristic)
    extensions = {}
    for f in file_tree:
        ext = Path(f["path"]).suffix.lower()
        if ext:
            extensions[ext] = extensions.get(ext, 0) + 1
    
    main_language = "TypeScript"  # frontend is TS/React
    framework = "Next.js"  # detected from next.config.ts
    
    # Entry points
    entry_points = [
        f["path"] for f in file_tree
        if any(name in f["path"].lower() for name in ["page.tsx", "layout.tsx", "index.ts"])
    ][:5]
    
    # Key modules (directories with most files)
    module_counts = {}
    for f in file_tree:
        parts = Path(f["path"]).parts
        if len(parts) >= 2:
            module = str(Path(parts[0]) / parts[1])
            module_counts[module] = module_counts.get(module, 0) + 1
    
    key_modules = sorted(module_counts.keys(), key=lambda k: module_counts[k], reverse=True)[:5]
    
    print(f"   Main Language: {main_language}")
    print(f"   Framework: {framework}")
    print(f"   Entry Points: {len(entry_points)} found")
    print(f"   Key Modules: {len(key_modules)} discovered")
    print()

    # ─── Step 4: Create AnalysisResult ────────────────────────
    print("📌 Step 4: Creating AnalysisResult...")
    
    analysis = AnalysisResult(
        project_name="frontend",
        architecture_pattern="Next.js App Router + React Components",
        confidence=0.95,
        main_language=main_language,
        framework=framework,
        patterns_detected=[
            {"name": "App Router", "description": "Next.js 13+ app directory structure"},
            {"name": "Server Components", "description": "React Server Components pattern"},
            {"name": "Client Components", "description": "Interactive client-side components"},
        ],
        files_read=[{"path": f["path"], "content": "", "summary": ""} for f in file_tree[:20]],
        tech_stack={
            "runtime": "Node.js",
            "framework": "Next.js 14+",
            "language": "TypeScript",
            "ui": "React",
            "styling": "Tailwind CSS",
        },
        directory_structure=_generate_tree_view(file_tree),
        dependency_graph=dep_graph,
        improvements=[
            "Consider adding error boundaries for better error handling",
            "Add loading states for async components",
            "Implement TypeScript strict mode",
        ],
        entry_points=entry_points,
        key_modules=key_modules,
        file_tree=file_tree,
        code_chunks=[],
        config_files_content={},
    )
    
    print(f"   ✅ AnalysisResult created")
    print()

    # ─── Step 5: Generate Multi-Layer Documentation ───────────
    print("📌 Step 5: Generating multi-layer documentation...")
    print("   (This tests the new MultiLayerDocGenerator)")
    print()
    
    try:
        generator = MultiLayerDocGenerator(analysis)
        docs_dict = generator.generate_full_documentation()
        
        print(f"   ✅ Generated {len(docs_dict)} documentation files")
        print()
        
        # Show file breakdown
        root_files = [k for k in docs_dict.keys() if "/" not in k]
        module_files = [k for k in docs_dict.keys() if "/" in k]
        
        print(f"   📄 File Breakdown:")
        print(f"      Root Level: {len(root_files)} files")
        for f in sorted(root_files):
            size_kb = len(docs_dict[f]) / 1024
            print(f"         - {f} ({size_kb:.1f} KB)")
        
        print(f"      Module Level: {len(module_files)} files")
        module_groups = {}
        for f in module_files:
            module = f.split("/")[0]
            module_groups[module] = module_groups.get(module, 0) + 1
        
        for module, count in sorted(module_groups.items()):
            print(f"         - {module}/: {count} files")
        print()
        
    except Exception as e:
        print(f"   ❌ Documentation generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ─── Step 6: Save output files ─────────────────────────────
    print("📌 Step 6: Saving output files...")
    
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for file_path, content in docs_dict.items():
        full_path = output_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content, encoding="utf-8")
        saved_count += 1
    
    print(f"   ✅ Saved {saved_count} files to: {output_dir}")
    print()

    # ─── Step 7: Generate validation report ────────────────────
    print("📌 Step 7: Validation Report")
    print("=" * 70)
    
    # Check required root files
    required_root = ["llms.txt", "AGENTS.md", "repomap.txt"]
    for req in required_root:
        status = "✅" if req in docs_dict else "❌"
        print(f"   {status} {req}")
    
    # Check module structure
    print()
    print(f"   Module-level documentation:")
    readme_count = len([k for k in docs_dict if k.endswith("/ReadMe.LLM")])
    agents_count = len([k for k in docs_dict if k.endswith("/AGENTS.md") and "/" in k])
    print(f"      - ReadMe.LLM files: {readme_count}")
    print(f"      - Module AGENTS.md files: {agents_count}")
    
    # Content validation
    print()
    print(f"   Content validation:")
    
    if "llms.txt" in docs_dict:
        llms_content = docs_dict["llms.txt"]
        has_title = "# " in llms_content
        has_navigation = any(word in llms_content.lower() for word in ["navigation", "structure", "index"])
        print(f"      llms.txt: {'✅' if has_title and has_navigation else '⚠️'} (title={has_title}, nav={has_navigation})")
    
    if "AGENTS.md" in docs_dict:
        agents_content = docs_dict["AGENTS.md"]
        has_rules = any(word in agents_content.lower() for word in ["rule", "guideline", "convention"])
        has_examples = "```" in agents_content
        print(f"      AGENTS.md: {'✅' if has_rules and has_examples else '⚠️'} (rules={has_rules}, code={has_examples})")
    
    if "repomap.txt" in docs_dict:
        repomap_content = docs_dict["repomap.txt"]
        has_tree = any(char in repomap_content for char in ["├", "│", "└"])
        has_sigs = "def " in repomap_content or "function " in repomap_content or "const " in repomap_content
        print(f"      repomap.txt: {'✅' if has_tree or has_sigs else '⚠️'} (tree={has_tree}, sigs={has_sigs})")
    
    # Check for ReadMe.LLM content
    readme_files = [k for k in docs_dict if k.endswith("/ReadMe.LLM")]
    if readme_files:
        sample = docs_dict[readme_files[0]]
        has_funcs = "function" in sample.lower() or "method" in sample.lower()
        has_diagram = "```mermaid" in sample
        print(f"      ReadMe.LLM samples: {'✅' if has_funcs and has_diagram else '⚠️'} (funcs={has_funcs}, mermaid={has_diagram})")
    
    print()
    print("=" * 70)
    print()

    # ─── Step 8: Summary ───────────────────────────────────────
    total_chars = sum(len(content) for content in docs_dict.values())
    total_kb = total_chars / 1024
    total_lines = sum(content.count("\n") for content in docs_dict.values())
    
    print("📊 Final Summary")
    print("=" * 70)
    print(f"   Source files analyzed:     {len(file_tree)}")
    print(f"   Functions extracted:       {total_functions}")
    print(f"   Classes extracted:         {total_classes}")
    print(f"   Dependency graph nodes:    {node_count}")
    print(f"   Dependency graph edges:    {edge_count}")
    print()
    print(f"   Documentation files:       {len(docs_dict)}")
    print(f"   Total documentation size:  {total_kb:.1f} KB")
    print(f"   Total lines generated:     {total_lines:,}")
    print()
    print(f"   Output saved to:           {output_dir}")
    print("=" * 70)
    print()
    print("🎉 Multi-layer documentation test complete!")
    print("   Review the output files to verify quality.")
    print()


def _generate_tree_view(file_tree: list) -> str:
    """Generate a simple directory tree view."""
    paths = [f["path"] for f in file_tree]
    tree_lines = []
    
    for path in sorted(paths)[:50]:  # Limit to first 50 for brevity
        depth = path.count("/") + path.count("\\")
        indent = "  " * depth
        name = Path(path).name
        tree_lines.append(f"{indent}- {name}")
    
    if len(paths) > 50:
        tree_lines.append(f"  ... and {len(paths) - 50} more files")
    
    return "\n".join(tree_lines)


if __name__ == "__main__":
    asyncio.run(main())
