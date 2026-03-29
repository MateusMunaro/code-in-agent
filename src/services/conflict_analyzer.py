"""
Conflict Analyzer Service.

Orchestrates multi-branch conflict analysis:
1. Collects diffs between all branch pairs
2. Identifies overlapping files (direct conflict risk)
3. Uses LLM for deep semantic analysis (indirect conflict risk)
4. Generates actionable recommendations and merge order
"""

import json
import re
import asyncio
from itertools import combinations
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from ..llm.provider import MultiModelChat
from ..services.git_service import GitService
from ..graph.state import (
    ConflictAnalysisState,
    BranchDiff,
    ConflictRisk,
    create_conflict_analysis_state,
)


class ConflictAnalyzer:
    """
    Analyzes multiple branches for potential merge conflicts.

    Goes beyond simple file-level overlap detection by using LLM
    to understand the semantic context of changes across branches,
    detecting indirect conflicts (e.g., branch A changes an API
    signature that branch B depends on).
    """

    # Maximum diff content to send to LLM per branch pair
    MAX_DIFF_CONTEXT = 8000
    # Maximum file content to send for semantic analysis
    MAX_FILE_CONTEXT = 4000

    def __init__(self, chat: MultiModelChat, git_service: GitService):
        self.chat = chat
        self.git_service = git_service

    def log(self, message: str):
        print(f"[ConflictAnalyzer] {message}")

    def _parse_llm_json(self, text: str) -> dict:
        """Parse JSON from LLM response, stripping markdown fences."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\w*\s*', '', text)
        text = re.sub(r'```', '', text)
        text = text.strip()
        return json.loads(text)

    async def analyze(
        self,
        repo_url: str,
        repo_path: str,
        branches: list[str],
        job_id: str,
        model_id: str = None,
    ) -> ConflictAnalysisState:
        """
        Run the full conflict analysis pipeline.

        Args:
            repo_url: Repository URL
            repo_path: Path to the cloned repository (full clone)
            branches: List of branch names to analyze
            job_id: Job ID for tracking
            model_id: LLM model to use

        Returns:
            ConflictAnalysisState with all analysis results
        """
        state = create_conflict_analysis_state(
            job_id=job_id,
            repo_url=repo_url,
            repo_path=repo_path,
            branches=branches,
        )

        try:
            # Step 1: Validate branches exist
            self.log(f"Validating {len(branches)} branches...")
            available = await self.git_service.list_remote_branches(repo_path)
            missing = [b for b in branches if b not in available]
            if missing:
                state["error"] = f"Branches not found: {missing}. Available: {available}"
                return state
            self.log(f"✅ All branches found")

            # Step 2: Compute diffs between all branch pairs
            self.log(f"Computing diffs for {len(list(combinations(branches, 2)))} branch pairs...")
            state = await self._compute_all_diffs(state)
            self.log(f"✅ Diffs computed: {len(state['branch_diffs'])} pairs")

            # Step 3: Identify overlapping files
            state = self._identify_overlapping_files(state)
            overlap_count = len(state["overlapping_files"])
            self.log(f"✅ Found {overlap_count} files modified by multiple branches")

            # Step 4: Semantic analysis per branch (understand intent)
            self.log("Running semantic analysis of branch changes...")
            state = await self._semantic_analysis_per_branch(state)
            self.log(f"✅ Semantic analysis done for {len(state['semantic_context'])} branches")

            # Step 5: LLM conflict detection (the core analysis)
            self.log("Running LLM conflict risk analysis...")
            state = await self._llm_conflict_analysis(state)
            self.log(f"✅ Found {len(state['conflict_risks'])} conflict risks")

            # Step 6: Generate merge order recommendation
            self.log("Generating merge order recommendation...")
            state = await self._generate_merge_recommendations(state)
            self.log(f"✅ Merge order: {state['merge_order_suggestion']}")

            state["confidence"] = min(0.95, 0.5 + 0.1 * len(state["branch_diffs"]))

        except Exception as e:
            self.log(f"❌ Analysis failed: {e}")
            state["error"] = str(e)

        return state

    async def _compute_all_diffs(
        self, state: ConflictAnalysisState
    ) -> ConflictAnalysisState:
        """Compute diffs between all pairs of branches."""
        branches = state["branches"]
        repo_path = state["repo_path"]

        # Compute diffs for all pairs in parallel
        tasks = []
        pairs = list(combinations(branches, 2))

        for branch_a, branch_b in pairs:
            tasks.append(
                self.git_service.get_diff_between_branches(
                    repo_path, branch_a, branch_b
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        diffs = []
        branch_files: dict[str, set[str]] = {b: set() for b in branches}

        for i, result in enumerate(results):
            branch_a, branch_b = pairs[i]

            if isinstance(result, Exception):
                self.log(f"⚠️ Diff {branch_a}...{branch_b} failed: {result}")
                continue

            diff = BranchDiff(
                branch_a=result["branch_a"],
                branch_b=result["branch_b"],
                files_changed=result["files_changed"],
                stats=result["stats"],
            )
            diffs.append(diff)

            # Track which files each branch modifies
            for file_change in result["files_changed"]:
                branch_files[branch_b].add(file_change["path"])

        # Also compute reverse direction to capture branch_a changes
        reverse_tasks = []
        for branch_a, branch_b in pairs:
            reverse_tasks.append(
                self.git_service.get_diff_between_branches(
                    repo_path, branch_b, branch_a
                )
            )

        reverse_results = await asyncio.gather(*reverse_tasks, return_exceptions=True)

        for i, result in enumerate(reverse_results):
            branch_a, branch_b = pairs[i]
            if isinstance(result, Exception):
                continue
            for file_change in result["files_changed"]:
                branch_files[branch_a].add(file_change["path"])

        state["branch_diffs"] = diffs
        state["branch_files"] = {b: list(files) for b, files in branch_files.items()}

        return state

    def _identify_overlapping_files(
        self, state: ConflictAnalysisState
    ) -> ConflictAnalysisState:
        """Find files modified by more than one branch."""
        file_to_branches: dict[str, list[str]] = {}

        for branch, files in state["branch_files"].items():
            for file_path in files:
                if file_path not in file_to_branches:
                    file_to_branches[file_path] = []
                file_to_branches[file_path].append(branch)

        # Only keep files touched by 2+ branches
        overlapping = {
            path: branches
            for path, branches in file_to_branches.items()
            if len(branches) >= 2
        }

        state["overlapping_files"] = overlapping
        return state

    async def _semantic_analysis_per_branch(
        self, state: ConflictAnalysisState
    ) -> ConflictAnalysisState:
        """
        Use LLM to understand the semantic intent of each branch's changes.

        This is critical for detecting INDIRECT conflicts — e.g., branch A
        changes an API contract that branch B depends on, even though they
        don't modify the same files.
        """
        tasks = []
        branch_names = []

        for branch in state["branches"]:
            # Collect the diffs relevant to this branch
            branch_changes = []
            for diff in state["branch_diffs"]:
                if diff["branch_b"] == branch:
                    for fc in diff["files_changed"][:10]:  # Limit files
                        content = fc.get("diff_content", "")
                        if len(content) > self.MAX_FILE_CONTEXT:
                            content = content[:self.MAX_FILE_CONTEXT] + "\n... (truncated)"
                        branch_changes.append({
                            "path": fc["path"],
                            "change_type": fc["change_type"],
                            "diff": content,
                        })

            if not branch_changes:
                continue

            branch_names.append(branch)
            tasks.append(self._analyze_branch_intent(branch, branch_changes))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        semantic_context = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log(f"⚠️ Semantic analysis for {branch_names[i]} failed: {result}")
                semantic_context[branch_names[i]] = "Analysis failed"
            else:
                semantic_context[branch_names[i]] = result

        state["semantic_context"] = semantic_context
        return state

    async def _analyze_branch_intent(
        self, branch: str, changes: list[dict]
    ) -> str:
        """Analyze the semantic intent of a single branch's changes."""
        changes_text = ""
        for change in changes[:8]:
            changes_text += f"\n### {change['path']} ({change['change_type']})\n"
            changes_text += f"```diff\n{change['diff'][:2000]}\n```\n"

        messages = [
            SystemMessage(content="""You are an expert software engineer analyzing branch changes.
Analyze the diffs and provide a concise semantic summary of what this branch does.
Focus on:
1. What feature/fix is being implemented
2. Which APIs, interfaces, or contracts are being changed
3. Which modules/components are being affected
4. Any side effects that could impact other parts of the codebase

Respond in plain text (NOT JSON), 3-5 sentences maximum."""),
            HumanMessage(content=f"""Branch: {branch}

Changes:
{changes_text}

What is this branch doing semantically?"""),
        ]

        response = await self.chat.ainvoke(messages)
        return response.content.strip()

    async def _llm_conflict_analysis(
        self, state: ConflictAnalysisState
    ) -> ConflictAnalysisState:
        """
        Core LLM analysis: detect both direct and indirect conflicts.

        This is the most important step — the LLM receives:
        1. Semantic summaries of each branch
        2. Overlapping files (direct conflicts)
        3. Diffs between branches
        And produces conflict risks with recommendations.
        """
        # Build the context for the LLM
        semantic_summaries = ""
        for branch, summary in state["semantic_context"].items():
            semantic_summaries += f"\n### Branch `{branch}`\n{summary}\n"

        overlapping_info = ""
        if state["overlapping_files"]:
            overlapping_info = "\n## Files Modified by Multiple Branches (Direct Conflict Risk)\n"
            for path, branches in state["overlapping_files"].items():
                overlapping_info += f"- **{path}**: modified by {', '.join(branches)}\n"
        else:
            overlapping_info = "\n## No files are modified by multiple branches directly.\n"

        # Include key diffs for overlapping files
        conflict_diffs = ""
        if state["overlapping_files"]:
            conflict_diffs = "\n## Diffs for Overlapping Files\n"
            for diff in state["branch_diffs"]:
                for fc in diff["files_changed"]:
                    if fc["path"] in state["overlapping_files"]:
                        diff_content = fc.get("diff_content", "")[:2000]
                        conflict_diffs += (
                            f"\n### {fc['path']} (branch `{diff['branch_b']}` vs `{diff['branch_a']}`)\n"
                            f"```diff\n{diff_content}\n```\n"
                        )

        # Truncate if too long
        total_context = semantic_summaries + overlapping_info + conflict_diffs
        if len(total_context) > 15000:
            total_context = total_context[:15000] + "\n\n... (context truncated for size)"

        messages = [
            SystemMessage(content="""You are an expert software engineer specialized in merge conflict prevention.

You are analyzing multiple branches from the same repository to identify potential merge conflicts BEFORE they happen.

Your analysis must detect TWO types of conflicts:

1. **DIRECT CONFLICTS**: Multiple branches modify the same file in overlapping regions
2. **INDIRECT/SEMANTIC CONFLICTS**: Branches that don't touch the same files but can cause issues:
   - Branch A changes an API signature that Branch B calls
   - Branch A renames/moves a module that Branch B imports
   - Branch A changes a database schema that Branch B queries
   - Branch A changes config/env that Branch B depends on
   - Branch A changes shared types/interfaces used by Branch B
   - Logical contradictions between branches (e.g., opposite business rules)

For each conflict risk, provide:
- The specific file(s) involved
- Risk level (high/medium/low)
- Which branches are in conflict
- Clear description of WHY it's a conflict
- Actionable recommendation for the development team

Respond in JSON format:
{
    "conflict_risks": [
        {
            "file_path": "path/to/file",
            "risk_level": "high|medium|low",
            "conflicting_branches": ["branch-a", "branch-b"],
            "description": "Detailed description of the conflict",
            "recommendation": "How to resolve or prevent this conflict"
        }
    ],
    "general_recommendations": [
        "General advice for the team..."
    ],
    "merge_order_suggestion": ["branch-1", "branch-2", "branch-3"]
}"""),
            HumanMessage(content=f"""## Repository: {state['repo_url']}
## Branches being analyzed: {', '.join(state['branches'])}

## Semantic Summary of Each Branch's Changes
{semantic_summaries}

{overlapping_info}

{conflict_diffs}

## Branch Change Statistics
{json.dumps({b: len(files) for b, files in state['branch_files'].items()}, indent=2)}

Analyze all potential conflicts (direct AND indirect/semantic) and provide your recommendations."""),
        ]

        try:
            response = await self.chat.ainvoke(messages)
            result = self._parse_llm_json(response.content)

            # Parse conflict risks
            for risk_data in result.get("conflict_risks", []):
                risk = ConflictRisk(
                    file_path=risk_data.get("file_path", ""),
                    risk_level=risk_data.get("risk_level", "medium"),
                    conflicting_branches=risk_data.get("conflicting_branches", []),
                    description=risk_data.get("description", ""),
                    recommendation=risk_data.get("recommendation", ""),
                )
                state["conflict_risks"].append(risk)

            state["general_recommendations"] = result.get("general_recommendations", [])
            state["merge_order_suggestion"] = result.get("merge_order_suggestion", [])

        except Exception as e:
            self.log(f"LLM conflict analysis failed: {e}")
            # Fallback: report overlapping files as risks
            for path, branches in state["overlapping_files"].items():
                state["conflict_risks"].append(ConflictRisk(
                    file_path=path,
                    risk_level="high",
                    conflicting_branches=branches,
                    description=f"File modified by multiple branches: {', '.join(branches)}",
                    recommendation="Coordinate merge order to avoid conflicts in this file.",
                ))

        return state

    async def _generate_merge_recommendations(
        self, state: ConflictAnalysisState
    ) -> ConflictAnalysisState:
        """
        Generate a final comprehensive report with merge order and strategy.
        Only runs if we don't already have a merge order from the main analysis.
        """
        if state["merge_order_suggestion"]:
            return state

        # Simple heuristic: branches with fewer conflicts should merge first
        branch_risk_count = {b: 0 for b in state["branches"]}
        for risk in state["conflict_risks"]:
            for branch in risk["conflicting_branches"]:
                if branch in branch_risk_count:
                    branch_risk_count[branch] += 1

        # Sort by risk count (ascending = merge safest first)
        sorted_branches = sorted(
            branch_risk_count.items(), key=lambda x: x[1]
        )
        state["merge_order_suggestion"] = [b for b, _ in sorted_branches]

        return state

    def build_result_summary(self, state: ConflictAnalysisState) -> str:
        """
        Build a human-readable markdown summary of the conflict analysis.

        Returns:
            Markdown formatted summary
        """
        parts = []

        parts.append(f"# 🔀 Branch Conflict Analysis Report\n")
        parts.append(f"**Repository**: {state['repo_url']}")
        parts.append(f"**Branches analyzed**: {', '.join(state['branches'])}")
        parts.append(f"**Confidence**: {state['confidence']:.0%}\n")

        # Branch summaries
        if state["semantic_context"]:
            parts.append("## 📋 Branch Summaries\n")
            for branch, summary in state["semantic_context"].items():
                parts.append(f"### `{branch}`\n{summary}\n")

        # Conflict risks
        if state["conflict_risks"]:
            parts.append("## ⚠️ Conflict Risks\n")

            # Group by risk level
            for level in ["high", "medium", "low"]:
                level_risks = [
                    r for r in state["conflict_risks"]
                    if r["risk_level"] == level
                ]
                if level_risks:
                    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[level]
                    parts.append(f"### {emoji} {level.upper()} Risk\n")
                    for risk in level_risks:
                        parts.append(f"**File**: `{risk['file_path']}`")
                        parts.append(f"**Branches**: {', '.join(risk['conflicting_branches'])}")
                        parts.append(f"**Issue**: {risk['description']}")
                        parts.append(f"**Recommendation**: {risk['recommendation']}\n")
        else:
            parts.append("## ✅ No Conflict Risks Detected\n")
            parts.append("The analyzed branches appear to have no conflicting changes.\n")

        # Merge order
        if state["merge_order_suggestion"]:
            parts.append("## 📐 Recommended Merge Order\n")
            for i, branch in enumerate(state["merge_order_suggestion"], 1):
                parts.append(f"{i}. `{branch}`")
            parts.append("")

        # General recommendations
        if state["general_recommendations"]:
            parts.append("## 💡 General Recommendations\n")
            for rec in state["general_recommendations"]:
                parts.append(f"- {rec}")
            parts.append("")

        # Statistics
        parts.append("## 📊 Statistics\n")
        for branch, files in state["branch_files"].items():
            parts.append(f"- `{branch}`: {len(files)} files modified")

        if state["overlapping_files"]:
            parts.append(f"\n**Overlapping files**: {len(state['overlapping_files'])}")
            for path, branches in state["overlapping_files"].items():
                parts.append(f"  - `{path}` → {', '.join(branches)}")

        return "\n".join(parts)
