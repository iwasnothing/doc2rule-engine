"""
Review an execution report produced by 05_execute_rules.py using an LLM.

Usage:
    python 06_review_report.py <execution_report.json> [--rules-file <rules_cel.json>] [--chunk-size 50]

The script will:
  1. Load the execution report (output of 05_execute_rules.py).
  2. Optionally load the original enriched rules for additional context.
  3. Pre-compute global statistics (health, input/output variable
     distributions, path analysis) from the full report.
  4. Chunk the run records into batches (default 50 runs per chunk)
     to stay within the LLM context window.
  5. For each chunk, send global stats + that chunk's run records to
     the LLM for a partial review.
  6. Send all chunk reviews to the LLM in a final aggregation call
     to produce the combined report: data analysed, outputs,
     conclusions & insights, correctness, and confidence score.
  7. Save as JSON and Markdown.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# LLM config from environment
OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")


def _get_llm() -> ChatOpenAI:
    """Create ChatOpenAI instance from environment config."""
    kwargs: dict[str, Any] = {"model": MODEL_NAME, "temperature": 0}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    return ChatOpenAI(**kwargs)


# ============================================================
# PART 1: Pydantic models
# ============================================================

class Issue(BaseModel):
    """A single issue or anomaly found."""
    severity: str = Field(
        ..., description="One of: 'critical', 'warning', 'info'."
    )
    description: str = Field(
        ..., description="Concise description of the issue."
    )
    affected_rules: List[str] = Field(
        default_factory=list,
        description="Rule IDs affected by this issue.",
    )


# -- Chunk-level review (partial) --

class ChunkReview(BaseModel):
    """LLM review of one chunk of execution runs."""
    chunk_index: int = Field(..., description="Which chunk this is (0-based).")
    runs_reviewed: int = Field(..., description="Number of runs in this chunk.")
    data_sources_observed: List[str] = Field(
        ...,
        description="Data source prefixes seen in this chunk (e.g. ['EPS', 'SIS']).",
    )
    output_variables_observed: List[str] = Field(
        ...,
        description="Derived output variables computed in this chunk.",
    )
    summary: str = Field(
        ...,
        description=(
            "3-5 sentence summary: what data was analysed in this chunk, "
            "what outputs were produced, any errors or anomalies observed."
        ),
    )
    issues: List[Issue] = Field(
        default_factory=list,
        description="Issues found in this chunk.",
    )
    error_runs: int = Field(
        ..., description="Number of runs with ERROR outputs in this chunk."
    )
    null_output_runs: int = Field(
        ..., description="Number of runs producing null derived outputs in this chunk."
    )
    chunk_confidence: int = Field(
        ...,
        description="Confidence score for this chunk, 0-100.",
    )


# -- Final combined review --

class ExecutionReview(BaseModel):
    """Final combined LLM review of the full execution report."""

    # --- Section 1: Data Analysed ---
    data_analysed_summary: str = Field(
        ...,
        description=(
            "3-5 sentence summary of what input data was fed into the rule "
            "engine. Cover: data sources, key variable names, types of values, "
            "and scope of the test data."
        ),
    )
    input_data_sources: List[str] = Field(
        ...,
        description="Distinct data source prefixes (e.g. ['EPS', 'SIS']).",
    )

    # --- Section 2: Outputs ---
    outputs_summary: str = Field(
        ...,
        description=(
            "3-5 sentence summary of what the rule engine produced. Cover: "
            "derived output variables, types of values, how outputs varied."
        ),
    )
    key_output_variables: List[str] = Field(
        ...,
        description="Most important derived output variables.",
    )

    # --- Section 3: Conclusions & Data Insights ---
    conclusions: str = Field(
        ...,
        description=(
            "3-5 sentence high-level conclusion about what the rule engine "
            "is doing end-to-end and whether the rules fulfil their purpose."
        ),
    )
    data_insights: List[str] = Field(
        ...,
        description="3-8 specific data insights or patterns observed.",
    )

    # --- Section 4: Correctness Evaluation ---
    correctness_evaluation: str = Field(
        ...,
        description=(
            "3-5 sentence evaluation of overall correctness. Address: "
            "error rate, null rate, routing logic, data consistency."
        ),
    )
    issues: List[Issue] = Field(
        default_factory=list,
        description="All issues found across all chunks, deduplicated.",
    )
    strengths: List[str] = Field(
        ..., description="2-5 things that worked well."
    )
    recommendations: List[str] = Field(
        ..., description="2-5 actionable recommendations."
    )

    # --- Final Score ---
    confidence_score: int = Field(
        ...,
        description=(
            "Overall confidence score, 0-100. "
            "100 = fully correct, 0 = completely broken."
        ),
    )


# ============================================================
# PART 2: Build global statistics & chunked run records
# ============================================================

def build_global_stats(
    report: Dict[str, Any],
    rule_repo: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Pre-compute global statistics from the full execution report.

    This is small enough to include in every chunk call as shared context.
    """
    results = report.get("results", [])
    total_runs = len(results)

    # Metadata
    meta = {
        "rules_file": report.get("rules_file"),
        "mockdata_file": report.get("mockdata_file"),
        "starting_rules_count": report.get("starting_rules_count"),
        "total_runs": report.get("total_runs", total_runs),
        "total_steps": report.get("total_steps"),
    }

    # Input variable summary
    all_input_vars: Counter[str] = Counter()
    input_value_samples: Dict[str, List[Any]] = defaultdict(list)
    for run in results:
        flat_init = _flatten(run.get("initial_data", {}))
        for var, val in flat_init.items():
            all_input_vars[var] += 1
            if len(input_value_samples[var]) < 5:
                input_value_samples[var].append(val)

    input_variables = []
    for var, count in all_input_vars.most_common(30):
        samples = list(dict.fromkeys(str(s) for s in input_value_samples.get(var, [])))[:5]
        input_variables.append({"variable": var, "seen_in_runs": count, "sample_values": samples})

    # Output / derived variable summary
    derived_vars: Counter[str] = Counter()
    derived_value_dist: Dict[str, Counter[str]] = defaultdict(Counter)
    error_count = 0
    null_output_count = 0
    for run in results:
        for var, val in run.get("context_variables", {}).items():
            if var.startswith("derived."):
                derived_vars[var] += 1
                derived_value_dist[var][str(val) if val is not None else "null"] += 1
                if isinstance(val, str) and val.startswith("ERROR"):
                    error_count += 1
                if val is None:
                    null_output_count += 1

    output_variables = []
    for var, count in derived_vars.most_common(30):
        dist = derived_value_dist[var].most_common(8)
        output_variables.append({
            "variable": var,
            "total": count,
            "distribution": {v: c for v, c in dist},
        })

    # Path analysis
    path_counter: Counter[str] = Counter()
    step_counts: List[int] = []
    terminal_states: Counter[str] = Counter()
    runs_with_errors = 0
    for run in results:
        trace = run.get("trace", [])
        step_counts.append(run.get("total_steps", len(trace)))
        path_ids = [s.get("rule_id", "?") for s in trace]
        path_counter[" -> ".join(path_ids)] += 1
        has_error = False
        for step in trace:
            cv = step.get("calculated_value")
            if isinstance(cv, str) and cv.startswith("ERROR"):
                has_error = True
            if step.get("status") == "unknown_or_terminal":
                terminal_states[step.get("rule_id", "?")] += 1
        if has_error:
            runs_with_errors += 1
        if trace:
            last = trace[-1]
            if last.get("next_rule") is None and "status" not in last:
                terminal_states[f"{last.get('rule_id', '?')} (no routing)"] += 1

    path_analysis = {
        "unique_paths": len(path_counter),
        "top_paths": [{"path": p, "count": c} for p, c in path_counter.most_common(15)],
        "steps": {
            "min": min(step_counts) if step_counts else 0,
            "max": max(step_counts) if step_counts else 0,
            "avg": round(sum(step_counts) / len(step_counts), 1) if step_counts else 0,
        },
        "terminal_states": [{"state": s, "count": c} for s, c in terminal_states.most_common(15)],
    }

    health = {
        "total_runs": total_runs,
        "runs_with_errors": runs_with_errors,
        "error_rate_pct": round(runs_with_errors / total_runs * 100, 1) if total_runs else 0,
        "total_error_outputs": error_count,
        "total_null_outputs": null_output_count,
        "null_output_rate_pct": round(
            null_output_count / max(sum(derived_vars.values()), 1) * 100, 1
        ),
    }

    return {
        "metadata": meta,
        "input_variables": input_variables,
        "output_variables": output_variables,
        "path_analysis": path_analysis,
        "health_statistics": health,
    }


def build_run_records(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert full run results into compact run records."""
    records = []
    for run in results:
        trace = run.get("trace", [])
        path = [s.get("rule_id", "?") for s in trace]
        updates = []
        for step in trace:
            for upd in step.get("context_updates", []):
                updates.append(upd)
        derived = {
            k: v for k, v in run.get("context_variables", {}).items()
            if k.startswith("derived.")
        }
        records.append({
            "start_rule": run.get("start_rule"),
            "row": run.get("data_row_index"),
            "steps": run.get("total_steps"),
            "path": path,
            "context_updates": updates,
            "derived_outputs": derived,
        })
    return records


def get_rule_defs_for_runs(
    run_records: List[Dict[str, Any]],
    rule_repo: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract compact rule definitions for only the rules visited in the runs."""
    visited: set[str] = set()
    for rec in run_records:
        for rid in rec.get("path", []):
            visited.add(rid)
    defs = []
    for rid in sorted(visited):
        r = rule_repo.get(rid)
        if r:
            defs.append({
                "rule_id": rid,
                "rule_name": r.get("rule_name", ""),
                "action": r.get("action", ""),
                "outcome": r.get("outcome", ""),
                "is_final": r.get("is_final", False),
            })
    return defs


def _flatten(data: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict into dotted-path keys."""
    flat: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(_flatten(value, full_key))
            else:
                flat[full_key] = value
    return flat


# ============================================================
# PART 3: LLM chunk review & final aggregation
# ============================================================

CHUNK_REVIEW_SYSTEM = """\
### SYSTEM ROLE
You are a Senior Data Analyst reviewing a CHUNK of execution runs from \
a CEL (Common Expression Language) business rule engine.

### WHAT YOU RECEIVE
A JSON object with:
- `chunk_index`: Which chunk this is.
- `total_chunks`: How many chunks in total.
- `global_stats`: Pre-computed statistics for the FULL execution \
report (metadata, health, input/output variable distributions, path \
analysis). Use this as context.
- `run_records`: The compact run records for THIS chunk only. Each has:
  - `start_rule`: Entry-point rule ID.
  - `path`: List of rule IDs visited.
  - `context_updates`: Variables written at each step.
  - `derived_outputs`: Final derived output values.
- `rule_definitions` (optional): What each visited rule does.

### YOUR TASK
Review this chunk and produce:
1. `data_sources_observed`: Data source prefixes seen.
2. `output_variables_observed`: Derived variables computed.
3. `summary`: 3-5 sentences on what data was analysed, outputs produced, \
and any errors or anomalies.
4. `issues`: Specific problems found (critical/warning/info).
5. `error_runs`: Count of runs with ERROR outputs.
6. `null_output_runs`: Count of runs with null derived outputs.
7. `chunk_confidence`: Confidence score 0-100."""


def review_chunk(
    chunk_index: int,
    total_chunks: int,
    global_stats: Dict[str, Any],
    run_records: List[Dict[str, Any]],
    rule_defs: Optional[List[Dict[str, Any]]] = None,
) -> ChunkReview:
    """Send one chunk of runs to the LLM for review."""
    llm = _get_llm()

    payload = {
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "global_stats": global_stats,
        "run_records": run_records,
    }
    if rule_defs:
        payload["rule_definitions"] = rule_defs

    user_input = json.dumps(payload, indent=2, default=str)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHUNK_REVIEW_SYSTEM),
        ("human", "{input}"),
    ])

    chain = prompt | llm.with_structured_output(ChunkReview)
    result = chain.invoke({"input": user_input})
    return result


FINAL_REVIEW_SYSTEM = """\
### SYSTEM ROLE
You are a Senior QA Director. You are combining multiple chunk reviews \
into a single executive review of a CEL rule engine execution.

### WHAT YOU RECEIVE
A JSON object with:
- `global_stats`: Full execution statistics (metadata, health, \
input/output variable distributions, path analysis).
- `chunk_reviews`: A list of partial reviews, each covering a chunk \
of execution runs with: summary, issues, error/null counts, confidence.

### YOUR TASK
Synthesize everything into a final review with:

#### Section 1: Data Analysed
- What data sources and variables were tested?
- How broad was the test data coverage?

#### Section 2: Outputs
- What derived variables were computed?
- What were the most common output values?
- Was there good branch coverage?

#### Section 3: Conclusions & Data Insights
- Business purpose of the rule engine.
- Patterns, anomalies, noteworthy distributions.

#### Section 4: Correctness Evaluation
- Error rate assessment.
- Null output assessment.
- Routing logic assessment.
- Consolidated issues list (deduplicated across chunks).
- Strengths and recommendations.
- **confidence_score** (0-100).

### SCORING GUIDANCE
- 90-100: Clean execution, minimal nulls, no errors.
- 70-89: Mostly correct, some nulls or minor dead-ends.
- 50-69: Significant issues — high null rate, some errors.
- 30-49: Many errors or mostly null/broken output.
- 0-29: Fundamentally broken."""


def combine_chunk_reviews(
    global_stats: Dict[str, Any],
    chunk_reviews: List[ChunkReview],
) -> ExecutionReview:
    """Send all chunk reviews to the LLM for final aggregation."""
    llm = _get_llm()

    payload = {
        "global_stats": global_stats,
        "chunk_reviews": [cr.model_dump() for cr in chunk_reviews],
    }

    user_input = json.dumps(payload, indent=2, default=str)

    prompt = ChatPromptTemplate.from_messages([
        ("system", FINAL_REVIEW_SYSTEM),
        ("human", "{input}"),
    ])

    chain = prompt | llm.with_structured_output(ExecutionReview)

    print("\nCombining chunk reviews into final report ...")
    result = chain.invoke({"input": user_input})
    return result


# ============================================================
# PART 4: Markdown report rendering
# ============================================================

def render_markdown_report(
    review: ExecutionReview,
    global_stats: Dict[str, Any],
) -> str:
    """Render the review as a Markdown document."""
    lines: List[str] = []
    meta = global_stats.get("metadata", {})
    health = global_stats.get("health_statistics", {})

    lines.append("# Rule Engine Execution Review Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Rules file:** `{meta.get('rules_file', 'N/A')}`")
    lines.append(f"**Mock-data file:** `{meta.get('mockdata_file', 'N/A')}`")
    lines.append(f"**Starting rules:** {meta.get('starting_rules_count', 'N/A')}")
    lines.append(f"**Total runs:** {meta.get('total_runs', 'N/A')}")
    lines.append(f"**Total steps:** {meta.get('total_steps', 'N/A')}")
    lines.append("")

    # ── Confidence Score ──────────────────────────────────────────────────
    lines.append("---")
    lines.append(f"## Confidence Score: **{review.confidence_score} / 100**")
    lines.append("")

    # ── Health Statistics ─────────────────────────────────────────────────
    lines.append("### Health Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total runs | {health.get('total_runs', 'N/A')} |")
    lines.append(f"| Runs with errors | {health.get('runs_with_errors', 'N/A')} |")
    lines.append(f"| Error rate | {health.get('error_rate_pct', 'N/A')}% |")
    lines.append(f"| Total ERROR outputs | {health.get('total_error_outputs', 'N/A')} |")
    lines.append(f"| Total null outputs | {health.get('total_null_outputs', 'N/A')} |")
    lines.append(f"| Null output rate | {health.get('null_output_rate_pct', 'N/A')}% |")
    lines.append("")

    # ── Section 1: Data Analysed ──────────────────────────────────────────
    lines.append("---")
    lines.append("## 1. Data Analysed")
    lines.append("")
    lines.append(review.data_analysed_summary)
    lines.append("")
    if review.input_data_sources:
        lines.append("**Data sources:** " + ", ".join(f"`{s}`" for s in review.input_data_sources))
        lines.append("")

    # ── Section 2: Outputs ────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 2. Outputs")
    lines.append("")
    lines.append(review.outputs_summary)
    lines.append("")
    if review.key_output_variables:
        lines.append("**Key output variables:**")
        for v in review.key_output_variables:
            lines.append(f"- `{v}`")
        lines.append("")

    # Output value distributions from global stats
    output_vars = global_stats.get("output_variables", [])
    if output_vars:
        lines.append("### Output Value Distributions")
        lines.append("")
        for ov in output_vars[:20]:
            lines.append(f"**`{ov['variable']}`** ({ov['total']} computations)")
            lines.append("")
            lines.append("| Value | Count |")
            lines.append("|-------|-------|")
            for val, cnt in ov.get("distribution", {}).items():
                lines.append(f"| {val} | {cnt} |")
            lines.append("")

    # ── Section 3: Conclusions & Data Insights ────────────────────────────
    lines.append("---")
    lines.append("## 3. Conclusions & Data Insights")
    lines.append("")
    lines.append(review.conclusions)
    lines.append("")
    if review.data_insights:
        lines.append("### Key Insights")
        lines.append("")
        for insight in review.data_insights:
            lines.append(f"- {insight}")
        lines.append("")

    # ── Section 4: Correctness Evaluation ─────────────────────────────────
    lines.append("---")
    lines.append("## 4. Correctness Evaluation")
    lines.append("")
    lines.append(review.correctness_evaluation)
    lines.append("")

    if review.issues:
        lines.append("### Issues Found")
        lines.append("")
        lines.append("| Severity | Description | Affected Rules |")
        lines.append("|----------|-------------|----------------|")
        for iss in review.issues:
            rules_str = ", ".join(iss.affected_rules) if iss.affected_rules else "—"
            lines.append(f"| {iss.severity} | {iss.description} | {rules_str} |")
        lines.append("")

    if review.strengths:
        lines.append("### Strengths")
        lines.append("")
        for s in review.strengths:
            lines.append(f"- {s}")
        lines.append("")

    if review.recommendations:
        lines.append("### Recommendations")
        lines.append("")
        for r in review.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    # ── Appendix: Execution Path Statistics ───────────────────────────────
    path_analysis = global_stats.get("path_analysis", {})
    if path_analysis:
        lines.append("---")
        lines.append("## Appendix: Execution Path Statistics")
        lines.append("")
        step_dist = path_analysis.get("steps", {})
        lines.append(f"- **Unique paths:** {path_analysis.get('unique_paths', 'N/A')}")
        lines.append(f"- **Steps per run:** min={step_dist.get('min')}, "
                      f"max={step_dist.get('max')}, avg={step_dist.get('avg')}")
        lines.append("")

        top_paths = path_analysis.get("top_paths", [])
        if top_paths:
            lines.append("### Top Execution Paths")
            lines.append("")
            lines.append("| Count | Path |")
            lines.append("|-------|------|")
            for tp in top_paths[:15]:
                lines.append(f"| {tp['count']} | {tp['path']} |")
            lines.append("")

        terminal = path_analysis.get("terminal_states", [])
        if terminal:
            lines.append("### Terminal States")
            lines.append("")
            lines.append("| State | Count |")
            lines.append("|-------|-------|")
            for ts in terminal:
                lines.append(f"| {ts['state']} | {ts['count']} |")
            lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Review an execution report from 05_execute_rules.py using an LLM.",
    )
    parser.add_argument(
        "execution_report",
        help="Path to the execution report JSON (output of 05_execute_rules.py).",
    )
    parser.add_argument(
        "--rules-file",
        default=None,
        help=(
            "Path to the enriched rules JSON for additional context. "
            "If omitted, uses the path from the execution report metadata."
        ),
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=50,
        help="Number of runs per LLM chunk (default: 50). Lower = safer for small context windows.",
    )
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if not os.path.isfile(args.execution_report):
        print(f"Error: File not found: {args.execution_report}", file=sys.stderr)
        sys.exit(1)

    # ── 1. Load execution report ──────────────────────────────────────────
    with open(args.execution_report, "r", encoding="utf-8") as f:
        report: Dict[str, Any] = json.load(f)

    results = report.get("results", [])
    print(f"Loaded execution report: {len(results)} run(s) from {args.execution_report}")

    # ── 2. Optionally load rule definitions ───────────────────────────────
    rule_repo: Optional[Dict[str, Dict[str, Any]]] = None
    rules_file_path = args.rules_file or report.get("rules_file")
    if rules_file_path and os.path.isfile(rules_file_path):
        with open(rules_file_path, "r", encoding="utf-8") as f:
            rules_list: list[dict] = json.load(f)
        rule_repo = {r["rule_id"]: r for r in rules_list if "rule_id" in r}
        print(f"Loaded {len(rule_repo)} rule definitions from: {rules_file_path}")
    else:
        print("No rules file found — reviewing without rule definitions.")

    # ── 3. Build global statistics ────────────────────────────────────────
    print("Building global statistics ...")
    global_stats = build_global_stats(report, rule_repo)
    health = global_stats["health_statistics"]

    print(f"  Total runs: {health['total_runs']}, "
          f"Errors: {health['runs_with_errors']} ({health['error_rate_pct']}%), "
          f"Null outputs: {health['total_null_outputs']} ({health['null_output_rate_pct']}%)")

    # ── 4. Build compact run records & chunk them ─────────────────────────
    run_records = build_run_records(results)
    chunk_size = args.chunk_size
    total_chunks = math.ceil(len(run_records) / chunk_size)

    print(f"\nSplitting {len(run_records)} runs into {total_chunks} chunk(s) "
          f"of ~{chunk_size} runs each.\n")

    # ── 5. Review each chunk via LLM ──────────────────────────────────────
    chunk_reviews: List[ChunkReview] = []

    for i in range(total_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(run_records))
        chunk = run_records[start:end]

        # Get rule definitions for only the rules visited in this chunk
        rule_defs = None
        if rule_repo:
            rule_defs = get_rule_defs_for_runs(chunk, rule_repo)

        print(f"[{i + 1}/{total_chunks}] Reviewing runs {start}-{end - 1} "
              f"({len(chunk)} runs, {len(rule_defs or [])} rule defs) ...")

        try:
            cr = review_chunk(i, total_chunks, global_stats, chunk, rule_defs)
            chunk_reviews.append(cr)
            print(f"  -> Chunk confidence: {cr.chunk_confidence}/100, "
                  f"errors: {cr.error_runs}, nulls: {cr.null_output_runs}, "
                  f"issues: {len(cr.issues)}")
        except Exception as e:
            print(f"  -> ERROR reviewing chunk {i}: {e}")

    if not chunk_reviews:
        print("\nNo chunks were successfully reviewed. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── 6. Final aggregation ──────────────────────────────────────────────
    try:
        review = combine_chunk_reviews(global_stats, chunk_reviews)
    except Exception as e:
        print(f"\nERROR during final aggregation: {e}", file=sys.stderr)
        # Fallback: build a minimal review from chunk data
        avg_conf = round(sum(cr.chunk_confidence for cr in chunk_reviews) / len(chunk_reviews))
        all_issues = [iss for cr in chunk_reviews for iss in cr.issues]
        review = ExecutionReview(
            data_analysed_summary="See chunk reviews for details. Final aggregation failed.",
            input_data_sources=list(set(
                src for cr in chunk_reviews for src in cr.data_sources_observed
            )),
            outputs_summary="See chunk reviews for details.",
            key_output_variables=list(set(
                v for cr in chunk_reviews for v in cr.output_variables_observed
            )),
            conclusions=f"Final aggregation LLM call failed: {e}",
            data_insights=["Aggregation failed — refer to individual chunk summaries."],
            correctness_evaluation="Unable to compute — aggregation failed.",
            issues=all_issues,
            strengths=["Chunk-level reviews completed successfully."],
            recommendations=["Retry the final aggregation or check LLM connectivity."],
            confidence_score=avg_conf,
        )

    # ── 7. Print summary to console ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("  EXECUTION REVIEW RESULTS")
    print(f"{'='*70}")
    print(f"  Total runs          : {health['total_runs']}")
    print(f"  Runs with errors    : {health['runs_with_errors']}")
    print(f"  Error rate          : {health['error_rate_pct']}%")
    print(f"  Null output rate    : {health['null_output_rate_pct']}%")
    print(f"  Issues found        : {len(review.issues)}")
    print(f"  Chunks reviewed     : {len(chunk_reviews)}")
    print(f"  {'─'*50}")
    print(f"  CONFIDENCE SCORE    : {review.confidence_score} / 100")
    print(f"{'='*70}")

    print(f"\n--- Data Analysed ---")
    print(review.data_analysed_summary)

    print(f"\n--- Outputs ---")
    print(review.outputs_summary)

    print(f"\n--- Conclusions ---")
    print(review.conclusions)

    print(f"\n--- Correctness ---")
    print(review.correctness_evaluation)

    if review.data_insights:
        print(f"\n--- Key Insights ---")
        for insight in review.data_insights:
            print(f"  - {insight}")

    # ── 8. Save outputs ──────────────────────────────────────────────────
    base, _ext = os.path.splitext(args.execution_report)
    idx = base.find("_execution")
    if idx != -1:
        base = base[:idx]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_output = f"{base}_review_{timestamp}.json"
    review_payload = {
        "generated_at": datetime.now().isoformat(),
        "execution_report": args.execution_report,
        "rules_file": rules_file_path,
        "confidence_score": review.confidence_score,
        "review": review.model_dump(),
        "chunk_reviews": [cr.model_dump() for cr in chunk_reviews],
        "global_stats": global_stats,
    }
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(review_payload, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nJSON review saved to: {json_output}")

    # Markdown report
    md_output = f"{base}_review_{timestamp}.md"
    md_content = render_markdown_report(review, global_stats)
    with open(md_output, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown review saved to: {md_output}")
