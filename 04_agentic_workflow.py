"""
Agentic Rule-Engine Workflow
============================

A LangGraph-based multi-agent workflow that executes business rules from
``rules.json`` by coordinating five specialised agents:

1. **Planning Agent** – receives the starting rule, initialises state.
2. **Data Agent**     – fetches mock data via ``get_data`` from
   ``03_generate_data.py`` and caches it in state.
3. **Rule Agent**     – evaluates conditions / actions / outcomes using an LLM
   assisted by an arithmetic tool.
4. **Report Agent**   – compiles a human-readable execution report.
5. **Review Agent**   – cross-checks the report against ``rules.json`` and
   produces a confidence score.

Environment variables (required):
    OPENAI_API_BASE   – custom OpenAI-compatible API base URL
    OPENAI_API_KEY    – API key for the endpoint
    MODEL_NAME        – model identifier (e.g. "gpt-4o")

Usage:
    python 04_agentic_workflow.py <rule_id>          # e.g. R001
    python 04_agentic_workflow.py R001 --verbose
"""

from __future__ import annotations

import json
import math
import operator
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
import dotenv
dotenv.load_dotenv()

# ── Local imports ─────────────────────────────────────────────────────────────
from importlib import import_module

_gen_data_mod = import_module("03_generate_data")
get_data = _gen_data_mod.get_data

# ── Load rules once at module level ──────────────────────────────────────────
_RULES_PATH = Path(__file__).parent / "rules.json"

def _load_rules() -> dict[str, dict]:
    """Return a dict keyed by rule_id."""
    with open(_RULES_PATH, encoding="utf-8") as f:
        rules_list: list[dict] = json.load(f)
    return {r["rule_id"]: r for r in rules_list}

RULES: dict[str, dict] = _load_rules()


# ══════════════════════════════════════════════════════════════════════════════
# State definition
# ══════════════════════════════════════════════════════════════════════════════

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state that flows through every node in the graph."""

    # Current rule object being evaluated
    rule_executing: dict

    # Which agent should run next (used by the router)
    agent_to_run: Literal[
        "planning_agent",
        "data_agent",
        "rule_agent",
        "report_agent",
        "review_agent",
        "done",
    ]

    # Accumulated data keyed by data-item name → list[dict]
    data_cache: dict[str, list[dict]]

    # Ordered log of every rule evaluation performed
    execution_log: list[dict]

    # Final compiled report (markdown string)
    report: str

    # Review result with confidence score
    review: str

    # LLM message history (used internally by tool-calling nodes)
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════════════════════
# LLM initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm() -> ChatOpenAI:
    api_base = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", ""))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("MODEL_NAME", "gpt-4o")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY must be set. "
            "Also set OPENAI_API_BASE and MODEL_NAME as needed."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=api_base or None,
        temperature=0,
    )


# ── Rate-limit retry wrapper ─────────────────────────────────────────────────

_RETRY_LIMIT = 5
_RETRY_WAIT_SECONDS = 90


def _invoke_with_retry(llm, messages, **kwargs):
    """Call ``llm.invoke(messages)`` with automatic retry on transient errors.

    Handles:
      - 429 RateLimitError from the OpenAI SDK
      - ValueError raised by langchain-openai for provider errors (e.g. 503)

    Retries up to ``_RETRY_LIMIT`` times, waiting ``_RETRY_WAIT_SECONDS``
    between attempts.  Re-raises the original exception if all retries are
    exhausted.
    """
    from openai import RateLimitError

    for attempt in range(1, _RETRY_LIMIT + 1):
        try:
            return llm.invoke(messages, **kwargs)
        except RateLimitError:
            label = "Rate-limit (429)"
        except ValueError as exc:
            # langchain-openai wraps provider errors (e.g. 503) in a
            # ValueError whose string/dict payload contains a 'code' field.
            err_str = str(exc)
            if "503" in err_str or "Provider error" in err_str:
                label = "Provider error (503)"
            else:
                raise  # unrelated ValueError – don't swallow it

        # Common retry handling for all transient errors
        if attempt == _RETRY_LIMIT:
            print(
                f"  ✖ {label} – attempt {attempt}/{_RETRY_LIMIT} – "
                f"all retries exhausted.  Raising.",
                file=sys.stderr,
            )
            raise
        print(
            f"  ⏳ {label} (attempt {attempt}/{_RETRY_LIMIT}). "
            f"Waiting {_RETRY_WAIT_SECONDS}s before retry …",
            file=sys.stderr,
        )
        time.sleep(_RETRY_WAIT_SECONDS)


# ══════════════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════════════

@tool
def fetch_data(data_item: str, num_rows: int = 5) -> list[dict]:
    """Fetch mock data rows for a given data-item key from the data schema.

    Parameters
    ----------
    data_item : str
        A key (or substring) matching an entry in data_schema.json.
    num_rows : int
        Number of mock rows to generate (default 5).

    Returns
    -------
    list[dict]
        Generated rows.
    """
    try:
        return get_data(data_item, num_rows)
    except KeyError as exc:
        return [{"error": str(exc)}]


@tool
def arithmetic(operation: str, numbers: list[float]) -> float | bool | str:
    """Perform an arithmetic or comparison operation on a list of numbers.

    Supported operations
    --------------------
    Arithmetic:
        sum, subtract, multiply, divide, average, modulo, power,
        min, max, abs (applied to first number), round (round first
        number to int), floor, ceil, count

    Comparison (returns bool):
        greater_than, greater_than_or_equal, less_than,
        less_than_or_equal, equal, not_equal

    Parameters
    ----------
    operation : str
        The operation name (case-insensitive).
    numbers : list[float]
        Operands. For binary comparisons the first two elements are used
        (a <op> b). For aggregates the full list is used.

    Returns
    -------
    float | bool | str
        The computed result, or an error string.
    """
    op = operation.strip().lower().replace(" ", "_")
    nums = [float(n) for n in numbers]

    if not nums:
        return "Error: no numbers provided"

    try:
        # ── Aggregates ────────────────────────────────────────────────────
        if op == "sum":
            return sum(nums)
        if op == "subtract":
            result = nums[0]
            for n in nums[1:]:
                result -= n
            return result
        if op == "multiply":
            result = nums[0]
            for n in nums[1:]:
                result *= n
            return result
        if op == "divide":
            result = nums[0]
            for n in nums[1:]:
                if n == 0:
                    return "Error: division by zero"
                result /= n
            return result
        if op == "average":
            return sum(nums) / len(nums)
        if op == "modulo":
            if len(nums) < 2 or nums[1] == 0:
                return "Error: modulo requires two numbers, divisor != 0"
            return nums[0] % nums[1]
        if op == "power":
            return nums[0] ** (nums[1] if len(nums) > 1 else 2)
        if op == "min":
            return min(nums)
        if op == "max":
            return max(nums)
        if op == "abs":
            return abs(nums[0])
        if op == "round":
            return round(nums[0])
        if op == "floor":
            return math.floor(nums[0])
        if op == "ceil":
            return math.ceil(nums[0])
        if op == "count":
            return float(len(nums))
        if op == "percentage":
            if len(nums) < 2 or nums[1] == 0:
                return "Error: percentage requires [numerator, denominator]"
            return (nums[0] / nums[1]) * 100.0

        # ── Comparisons (first two elements) ─────────────────────────────
        if len(nums) < 2:
            return f"Error: comparison '{op}' requires at least 2 numbers"
        a, b = nums[0], nums[1]

        if op in ("greater_than", "gt"):
            return a > b
        if op in ("greater_than_or_equal", "gte", "ge"):
            return a >= b
        if op in ("less_than", "lt", "smaller_than"):
            return a < b
        if op in ("less_than_or_equal", "lte", "le", "smaller_than_or_equal"):
            return a <= b
        if op in ("equal", "eq", "equal_to", "equals"):
            return a == b
        if op in ("not_equal", "ne", "neq", "not_equal_to"):
            return a != b

        return f"Error: unknown operation '{operation}'"
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [fetch_data, arithmetic]


# ══════════════════════════════════════════════════════════════════════════════
# Agent nodes
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Planning Agent ─────────────────────────────────────────────────────────

def planning_agent(state: AgentState) -> dict:
    """Initialise state for the starting rule and route to the data agent."""
    rule = state["rule_executing"]
    rule_id = rule["rule_id"]
    rule_name = rule["rule_name"]

    # Ensure containers exist
    data_cache = state.get("data_cache") or {}
    execution_log = state.get("execution_log") or []

    execution_log.append({
        "step": "planning",
        "rule_id": rule_id,
        "rule_name": rule_name,
        "message": f"Planning agent initialised execution for {rule_id}: {rule_name}. "
                   f"Data required: {rule.get('data_required', [])}",
    })

    return {
        "rule_executing": rule,
        "agent_to_run": "data_agent",
        "data_cache": data_cache,
        "execution_log": execution_log,
        "messages": [],
    }


# ── 2. Data Agent ────────────────────────────────────────────────────────────

# Groups of field names that represent the same logical identifier.
# Within each group, values are drawn from a shared pool so that a
# student_id in one table matches student_id in another.
_ID_FIELD_GROUPS: list[set[str]] = [
    {"student_id"},
    {"school_id", "home_school_id", "serving_school_id",
     "prior_year_school_id", "home_school_RCDTS"},
    {"district_id"},
    {"enrollment_id"},
    {"teacher_EIN", "employee_EIN",
     "principal_EIN_current_year", "principal_EIN_prior_year"},
    {"entity_id"},
]


def _is_placeholder(rows: list[dict]) -> bool:
    """Return True if *rows* is a 'no schema match' placeholder."""
    return (
        isinstance(rows, list)
        and len(rows) == 1
        and isinstance(rows[0], dict)
        and "note" in rows[0]
        and "No schema match" in str(rows[0].get("note", ""))
    )


def _harmonize_ids(data_cache: dict[str, list[dict]],
                   data_items: list[str]) -> None:
    """Align identifier fields across *data_items* in *data_cache* in-place.

    For each ID-field group (e.g. student_id, school_id family) we:
      1. Collect the first pool of values found in any data source.
      2. Distribute those values into every other data source that
         contains a field in the same group.

    This ensures that ``student_id`` in SIS matches ``student_id`` in
    the EL status table, and ``school_id`` in one file matches
    ``home_school_id`` / ``serving_school_id`` in another.
    """
    for group in _ID_FIELD_GROUPS:
        # 1. Build the value pool from the first source that has a field
        #    in this group.
        pool: list[str] = []
        for item in data_items:
            if item not in data_cache:
                continue
            rows = data_cache[item]
            if not isinstance(rows, list) or not rows or _is_placeholder(rows):
                continue
            for field in group:
                if field in rows[0]:
                    pool = [r[field] for r in rows if field in r]
                    break
            if pool:
                break

        if not pool:
            continue

        # 2. Distribute pooled values to all sources containing any
        #    field in this group.
        for item in data_items:
            if item not in data_cache:
                continue
            rows = data_cache[item]
            if not isinstance(rows, list) or not rows or _is_placeholder(rows):
                continue
            for i, row in enumerate(rows):
                for field in group:
                    if field in row:
                        row[field] = pool[i % len(pool)]


def data_agent(state: AgentState) -> dict:
    """Fetch all data_required items for the current rule into data_cache."""
    rule = state["rule_executing"]
    data_cache: dict[str, list[dict]] = dict(state.get("data_cache") or {})
    execution_log: list[dict] = list(state.get("execution_log") or [])

    data_items = rule.get("data_required", [])
    fetched_keys: list[str] = []
    missing_keys: list[str] = []
    newly_fetched_items: list[str] = []          # items fetched this round

    for item in data_items:
        if item in data_cache:
            # Check if previously cached entry was a "no match" placeholder
            if _is_placeholder(data_cache[item]):
                missing_keys.append(item)
                fetched_keys.append(f"{item} (cached – no match)")
            else:
                fetched_keys.append(f"{item} (cached)")
            continue
        try:
            rows = get_data(item, num_rows=15)
            data_cache[item] = rows
            fetched_keys.append(item)
            newly_fetched_items.append(item)
        except KeyError:
            # Try a shorter substring match: strip parenthetical and em-dash suffixes
            short_key = item.split("(")[0].strip().split("–")[0].strip()
            try:
                rows = get_data(short_key, num_rows=15)
                data_cache[item] = rows
                fetched_keys.append(f"{item} (matched via '{short_key}')")
                newly_fetched_items.append(item)
            except KeyError:
                data_cache[item] = [{"note": f"No schema match for '{item}'"}]
                missing_keys.append(item)
                fetched_keys.append(f"{item} (NO MATCH)")

    # ── Harmonize identifiers across ALL data items for this rule ────
    # We pass the full data_items list (including cached items) so that
    # newly fetched data picks up IDs from previously cached sources and
    # vice-versa.
    _harmonize_ids(data_cache, data_items)

    # Build an informative log message
    total = len(data_items)
    matched = total - len(missing_keys)
    msg = (
        f"Data agent fetched {matched}/{total} data items for "
        f"rule {rule['rule_id']}."
    )
    if missing_keys:
        msg += (
            f" ⚠ MISSING {len(missing_keys)} data source(s): "
            + ", ".join(missing_keys)
        )

    execution_log.append({
        "step": "data_fetch",
        "rule_id": rule["rule_id"],
        "data_fetched": fetched_keys,
        "missing_data": missing_keys,
        "message": msg,
    })

    return {
        "data_cache": data_cache,
        "execution_log": execution_log,
        "agent_to_run": "rule_agent",
    }


# ── 3. Rule Agent ────────────────────────────────────────────────────────────

def rule_agent(state: AgentState) -> dict:
    """Use the LLM (with arithmetic tool) to evaluate the current rule."""
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    rule = state["rule_executing"]
    data_cache = state.get("data_cache") or {}
    execution_log: list[dict] = list(state.get("execution_log") or [])

    # Build relevant data context for the LLM (exclude "no match" placeholders)
    relevant_data: dict[str, Any] = {}
    for item in rule.get("data_required", []):
        if item in data_cache and not _is_placeholder(data_cache[item]):
            relevant_data[item] = data_cache[item]

    # Detect which data sources are missing or have no real data
    missing_data: list[str] = []
    available_data_items: list[str] = []
    for item in rule.get("data_required", []):
        if item in data_cache and not _is_placeholder(data_cache[item]):
            available_data_items.append(item)
        else:
            missing_data.append(item)

    missing_data_instruction = ""
    if missing_data:
        missing_data_instruction = (
            "\n\n⚠️ MISSING DATA WARNING ⚠️\n"
            "The following required data sources are MISSING or unavailable:\n"
            + "\n".join(f"  - {d}" for d in missing_data)
            + "\n\nIMPORTANT rules for handling missing data:\n"
            "1. Do NOT fabricate, assume, or hallucinate values for missing data.\n"
            "2. If a condition REQUIRES missing data to be evaluated, mark that "
            "   condition as 'CANNOT EVALUATE – data missing'.\n"
            "3. If the rule's core calculation depends on missing data (e.g., the "
            "   rule needs assessment scores but none are available), the outcome "
            "   MUST be 'No Data Available' – NOT a calculated rate.\n"
            "4. When selecting the outgoing route:\n"
            "   - If an outgoing route exists for 'No Data Available' or similar, "
            "     choose that route.\n"
            "   - If no such route exists but the rule is NOT final, choose the "
            "     first available route but flag the outcome as incomplete.\n"
            "   - NEVER select a 'Rate calculated' route if the rate could not "
            "     actually be computed due to missing data.\n"
            "5. Set is_final to true if no meaningful forward progress can be made.\n"
        )

    system_prompt = (
        "You are a Rule Evaluation Agent in an education report-card system.\n"
        "Your job is to evaluate a business rule using the provided data.\n\n"
        "You have access to an 'arithmetic' tool for ALL mathematical operations "
        "(sum, subtract, multiply, divide, average, percentage, greater_than, "
        "less_than, equal, etc.). Use it whenever you need to compute or compare numbers. "
        "LLMs are weak at arithmetic, so ALWAYS delegate math to the tool.\n\n"
        "CRITICAL EVALUATION RULES:\n"
        "- Evaluate EVERY condition listed in the rule explicitly.\n"
        "- When a condition specifies a numeric threshold (e.g., 'enrollment >= 10'), "
        "  you MUST use the arithmetic tool to verify the comparison against actual "
        "  data values. Do NOT approximate or skip threshold checks.\n"
        "- If data contains only placeholder text (random strings) instead of numeric "
        "  values where numbers are expected, treat that data as MISSING.\n"
        "- A condition that fails means the rule's 'negative' outgoing route should "
        "  be selected (e.g., 'School is NOT eligible' instead of 'School is eligible').\n"
        "- Clearly distinguish between conditions that PASS, FAIL, or CANNOT BE EVALUATED.\n"
        f"{missing_data_instruction}\n"
        "After evaluating, you MUST output a JSON block (fenced with ```json) "
        "with exactly these keys:\n"
        "  - rule_id: the rule ID\n"
        "  - rule_name: the rule name\n"
        "  - condition_evaluation: object mapping each condition to PASS/FAIL/MISSING "
        "    with a brief explanation\n"
        "  - data_issues: list of any data quality issues encountered (missing data, "
        "    placeholder values, insufficient records, etc.) – empty list if none\n"
        "  - action_performed: what action was taken\n"
        "  - outcome: the result/outcome\n"
        "  - outgoing_route_chosen: which outgoing route was selected (the condition text)\n"
        "  - next_rule_id: the next rule ID to execute (or null if final)\n"
        "  - is_final: true/false\n"
        "  - reasoning: step-by-step reasoning\n"
    )

    # Build a separate section listing data with "no match" notes
    missing_data_section = ""
    if missing_data:
        missing_data_section = (
            f"\n## Missing / Unavailable Data Sources\n"
            f"The following data sources could NOT be loaded:\n"
            + "\n".join(f"- **{d}**" for d in missing_data)
            + "\n\nDo NOT fabricate data for these sources.\n"
        )

    user_prompt = (
        f"## Rule to Evaluate\n"
        f"```json\n{json.dumps(rule, indent=2)}\n```\n\n"
        f"## Available Data\n"
        f"```json\n{json.dumps(relevant_data, indent=2, default=str)}\n```\n"
        f"{missing_data_section}\n"
        "Please evaluate the rule conditions using ONLY the data above. "
        "Use the arithmetic tool for any calculations or numeric comparisons. "
        "Then determine the action outcome and select the appropriate outgoing route. "
        "If critical data is missing and the rule's core computation cannot be performed, "
        "set the outcome to 'No Data Available' and choose the most appropriate route "
        "(prefer a negative/incomplete route over a 'Rate calculated' route). "
        "Output your structured JSON result at the end."
    )

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Iterative tool-calling loop
    max_iterations = 10
    for _ in range(max_iterations):
        response = _invoke_with_retry(llm_with_tools, messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute each tool call
        from langchain_core.messages import ToolMessage

        for tc in response.tool_calls:
            tool_fn = {t.name: t for t in TOOLS}.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                messages.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc["id"])
                )
            else:
                messages.append(
                    ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"])
                )

    # Parse the LLM's final answer
    final_text = response.content if isinstance(response.content, str) else str(response.content)

    # Extract JSON block from the response
    eval_result = _extract_json_block(final_text)

    if eval_result is None:
        # Fallback: construct a minimal result
        eval_result = {
            "rule_id": rule["rule_id"],
            "rule_name": rule["rule_name"],
            "condition_evaluation": "LLM did not return structured JSON",
            "action_performed": rule.get("action", ""),
            "outcome": rule.get("outcome", ""),
            "outgoing_route_chosen": None,
            "next_rule_id": None,
            "is_final": rule.get("is_final", True),
            "reasoning": final_text,
        }

    # ── Determine next rule and routing ─────────────────────────────────
    next_rule_id = eval_result.get("next_rule_id")
    is_final = eval_result.get("is_final", rule.get("is_final", False))
    chosen_route_text = (eval_result.get("outgoing_route_chosen") or "").strip()

    outgoing_routes = rule.get("outgoing_routes") or []

    # Strategy 1: If the LLM already returned a valid next_rule_id that
    # exists in RULES, trust it.
    if next_rule_id and next_rule_id in RULES:
        pass  # keep as-is

    # Strategy 2: The LLM returned an outgoing_route_chosen text but no
    # (or invalid) next_rule_id.  Try to match the text against the
    # rule's outgoing_routes to find the correct next_rule.
    elif chosen_route_text and outgoing_routes:
        chosen_lower = chosen_route_text.lower()
        best_match = None
        for route in outgoing_routes:
            route_cond = (route.get("condition") or "").lower()
            # Exact or substring match
            if route_cond == chosen_lower or chosen_lower in route_cond or route_cond in chosen_lower:
                best_match = route
                break
        if best_match is not None:
            next_rule_id = best_match.get("next_rule")  # may be null → terminal
            eval_result["next_rule_id"] = next_rule_id
            eval_result["outgoing_route_chosen"] = best_match.get("condition", "")
            if next_rule_id is None:
                is_final = True
                eval_result["is_final"] = True

    # Strategy 3: LLM gave neither next_rule_id nor a matchable route text.
    # Check if the LLM signalled failure / negative outcome and pick the
    # matching negative route if one exists.
    elif next_rule_id is None and not is_final and outgoing_routes:
        outcome_lower = str(eval_result.get("outcome", "")).lower()
        reasoning_lower = str(eval_result.get("reasoning", "")).lower()

        # Look for a route whose condition matches a negative / terminal
        # signal from the LLM (e.g., "School is NOT eligible", or
        # "No Data Available").
        negative_keywords = ["not eligible", "not ", "no data", "insufficient",
                             "cannot", "missing", "fail", "ineligible"]
        llm_signals_negative = any(
            kw in outcome_lower or kw in reasoning_lower
            for kw in negative_keywords
        )

        matched_route = None
        if llm_signals_negative:
            # Prefer a null-route (terminal) or a route whose condition
            # contains negative language.
            for route in outgoing_routes:
                rc = (route.get("condition") or "").lower()
                if any(kw in rc for kw in negative_keywords):
                    matched_route = route
                    break
            # If no negative route found, check for any null-route
            if matched_route is None:
                for route in outgoing_routes:
                    if route.get("next_rule") is None:
                        matched_route = route
                        break

        if matched_route is not None:
            next_rule_id = matched_route.get("next_rule")
            eval_result["next_rule_id"] = next_rule_id
            eval_result["outgoing_route_chosen"] = matched_route.get("condition", "")
            if next_rule_id is None:
                is_final = True
                eval_result["is_final"] = True
        else:
            # Last resort: pick the first route with a non-null next_rule
            for route in outgoing_routes:
                if route.get("next_rule") is not None:
                    next_rule_id = route["next_rule"]
                    eval_result["next_rule_id"] = next_rule_id
                    eval_result["outgoing_route_chosen"] = route.get("condition", "")
                    eval_result["_routing_note"] = "Fallback: first available route (LLM gave no signal)"
                    break

    execution_log.append({
        "step": "rule_evaluation",
        **eval_result,
    })

    # Decide next agent
    if is_final or next_rule_id is None:
        agent_to_run = "report_agent"
        next_rule_obj = rule  # keep current for report
    else:
        next_rule_obj = RULES.get(next_rule_id, rule)
        agent_to_run = "data_agent"

    return {
        "rule_executing": next_rule_obj,
        "agent_to_run": agent_to_run,
        "data_cache": data_cache,
        "execution_log": execution_log,
    }


def _extract_json_block(text: str) -> dict | None:
    """Try to extract a JSON object from an LLM response."""
    import re

    # Try fenced code blocks first
    match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a bare JSON object
    match = re.search(r"\{[^{}]*\"rule_id\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try the most aggressive approach: find outermost { ... }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return None


# ── 4. Report Agent ──────────────────────────────────────────────────────────

def report_agent(state: AgentState) -> dict:
    """Generate a comprehensive execution report using the LLM."""
    llm = _build_llm()

    execution_log = state.get("execution_log") or []
    data_cache = state.get("data_cache") or {}

    # Summarise data cache keys and sample sizes
    data_summary = {
        k: f"{len(v)} rows" if isinstance(v, list) else str(v)
        for k, v in data_cache.items()
    }

    system_prompt = (
        "You are a Report Agent. Your task is to generate a clear, detailed, "
        "well-structured Markdown report of a rule-engine execution.\n\n"
        "The report MUST include:\n"
        "1. **Executive Summary** – one-paragraph overview of what was done.\n"
        "2. **Rules Executed** – for each rule: rule ID, name, conditions evaluated, "
        "   action performed, outcome, and which outgoing route was taken.\n"
        "3. **Data Considered** – which data sources were used and sample sizes.\n"
        "4. **Execution Flow** – step-by-step trace of the execution path.\n"
        "5. **Reasoning & Observations** – detailed reasoning at each step.\n"
        "6. **Final Outcome & Decision** – the ultimate result.\n"
    )

    user_prompt = (
        f"## Execution Log\n"
        f"```json\n{json.dumps(execution_log, indent=2, default=str)}\n```\n\n"
        f"## Data Sources Used\n"
        f"```json\n{json.dumps(data_summary, indent=2)}\n```\n\n"
        "Please generate the execution report now."
    )

    response = _invoke_with_retry(llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    report_text = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "report": report_text,
        "agent_to_run": "review_agent",
    }


# ── 5. Review Agent ──────────────────────────────────────────────────────────

def review_agent(state: AgentState) -> dict:
    """Review the report against rules.json and produce a confidence score."""
    llm = _build_llm()

    report = state.get("report", "")
    execution_log = state.get("execution_log") or []
    data_cache = state.get("data_cache") or {}

    # Gather only the rules that were executed
    executed_rule_ids = [
        entry["rule_id"]
        for entry in execution_log
        if entry.get("step") == "rule_evaluation" and "rule_id" in entry
    ]
    executed_rules = {
        rid: RULES[rid] for rid in executed_rule_ids if rid in RULES
    }

    system_prompt = (
        "You are a Review Agent. Your job is to audit the execution report "
        "against the original business rules AND the actual data that was used "
        "during evaluation.\n\n"
        "For each rule that was executed, verify:\n"
        "  - Were all conditions correctly evaluated against the actual data?\n"
        "  - Were the data values used in computations accurate and consistent "
        "    with the raw data provided?\n"
        "  - Was the correct action performed?\n"
        "  - Was the correct outcome derived?\n"
        "  - Was the correct outgoing route selected?\n\n"
        "Pay special attention to:\n"
        "  - Whether the data values referenced in the execution log match the "
        "    raw data cache.\n"
        "  - Whether arithmetic operations (sums, averages, comparisons) were "
        "    performed correctly on the actual data.\n"
        "  - Whether any data was misinterpreted or overlooked.\n\n"
        "At the end, provide:\n"
        "1. A **Data Verification** section confirming whether the data was "
        "   used correctly.\n"
        "2. A list of any issues, discrepancies, or concerns.\n"
        "3. An overall **Confidence Score** from 0 to 100 indicating how "
        "   confident you are that the execution was correct.\n"
        "4. A brief justification for the score.\n\n"
        "Format your response in Markdown."
    )

    user_prompt = (
        f"## Original Rules (for verification)\n"
        f"```json\n{json.dumps(executed_rules, indent=2, default=str)}\n```\n\n"
        f"## Actual Data Used (data_cache)\n"
        f"```json\n{json.dumps(data_cache, indent=2, default=str)}\n```\n\n"
        f"## Execution Report to Review\n"
        f"{report}\n\n"
        f"## Raw Execution Log\n"
        f"```json\n{json.dumps(execution_log, indent=2, default=str)}\n```\n\n"
        "Please review the report against both the rules and the actual data, "
        "and provide your assessment."
    )

    response = _invoke_with_retry(llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    review_text = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "review": review_text,
        "agent_to_run": "done",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════

def router(state: AgentState) -> str:
    """Route to the next agent based on state['agent_to_run']."""
    next_agent = state.get("agent_to_run", "done")
    if next_agent in (
        "planning_agent",
        "data_agent",
        "rule_agent",
        "report_agent",
        "review_agent",
    ):
        return next_agent
    return "done"


# ══════════════════════════════════════════════════════════════════════════════
# Build the graph
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planning_agent", planning_agent)
    graph.add_node("data_agent", data_agent)
    graph.add_node("rule_agent", rule_agent)
    graph.add_node("report_agent", report_agent)
    graph.add_node("review_agent", review_agent)

    # Entry point
    graph.set_entry_point("planning_agent")

    # After planning → route (always goes to data_agent)
    graph.add_conditional_edges(
        "planning_agent",
        router,
        {
            "data_agent": "data_agent",
        },
    )

    # After data_agent → route (always goes to rule_agent)
    graph.add_conditional_edges(
        "data_agent",
        router,
        {
            "rule_agent": "rule_agent",
        },
    )

    # After rule_agent → route to data_agent (next rule) or report_agent (final)
    graph.add_conditional_edges(
        "rule_agent",
        router,
        {
            "data_agent": "data_agent",
            "report_agent": "report_agent",
        },
    )

    # After report_agent → route to review_agent
    graph.add_conditional_edges(
        "report_agent",
        router,
        {
            "review_agent": "review_agent",
        },
    )

    # After review_agent → done
    graph.add_conditional_edges(
        "review_agent",
        router,
        {
            "done": END,
        },
    )

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the agentic rule-engine workflow.",
    )
    parser.add_argument(
        "rule_id",
        help="Starting rule ID (e.g. R001)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print state after every step.",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=10,
        help="Maximum number of rules to traverse before forcing stop (default 10).",
    )
    args = parser.parse_args()

    rule_id = args.rule_id.upper()
    if rule_id not in RULES:
        print(f"Error: rule '{rule_id}' not found in rules.json.", file=sys.stderr)
        print(f"Available: {', '.join(sorted(RULES.keys()))}", file=sys.stderr)
        sys.exit(1)

    starting_rule = RULES[rule_id]

    print(f"{'=' * 70}")
    print(f"  AGENTIC RULE-ENGINE WORKFLOW")
    print(f"  Starting rule: {rule_id} – {starting_rule['rule_name']}")
    print(f"{'=' * 70}\n")

    app = build_graph()

    initial_state: AgentState = {
        "rule_executing": starting_rule,
        "agent_to_run": "planning_agent",
        "data_cache": {},
        "execution_log": [],
        "report": "",
        "review": "",
        "messages": [],
    }

    # Stream execution to show progress
    rules_evaluated = 0
    for step_output in app.stream(initial_state, {"recursion_limit": 50}):
        for node_name, node_state in step_output.items():
            print(f"  ▸ {node_name} completed")
            if args.verbose:
                # Print a short summary
                if "execution_log" in node_state:
                    latest = node_state["execution_log"][-1] if node_state["execution_log"] else {}
                    if latest:
                        print(f"    Last log: {latest.get('message', latest.get('step', ''))}")
                if node_name == "rule_agent":
                    rules_evaluated += 1
                    if rules_evaluated >= args.max_rules:
                        print(f"\n  ⚠ Reached max-rules limit ({args.max_rules}). Stopping.")

    # Retrieve final state
    final_state = app.invoke(initial_state, {"recursion_limit": 50})

    # Print report
    report_text = final_state.get("report", "(no report generated)")
    review_text = final_state.get("review", "(no review generated)")

    print(f"\n{'=' * 70}")
    print("  EXECUTION REPORT")
    print(f"{'=' * 70}\n")
    print(report_text)

    print(f"\n{'=' * 70}")
    print("  REVIEW & CONFIDENCE SCORE")
    print(f"{'=' * 70}\n")
    print(review_text)

    # Save reports to markdown files
    base_dir = Path(__file__).parent
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = base_dir / f"execution_report_{rule_id}_{timestamp}.md"
    review_path = base_dir / f"review_report_{rule_id}_{timestamp}.md"

    report_path.write_text(report_text, encoding="utf-8")
    review_path.write_text(review_text, encoding="utf-8")

    # Export data_cache to JSON file
    data_cache = final_state.get("data_cache", {})
    data_cache_path = base_dir / f"data_cache_{rule_id}_{timestamp}.json"
    data_cache_path.write_text(
        json.dumps(data_cache, indent=2, default=str), encoding="utf-8"
    )

    # Also write a canonical data_cache.json (always overwritten with latest run)
    canonical_cache_path = base_dir / "data_cache.json"
    canonical_cache_path.write_text(
        json.dumps(data_cache, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{'=' * 70}")
    print(f"  Reports saved:")
    print(f"    Execution report → {report_path.name}")
    print(f"    Review report    → {review_path.name}")
    print(f"    Data cache       → {data_cache_path.name}")
    print(f"    Data cache (latest) → {canonical_cache_path.name}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
