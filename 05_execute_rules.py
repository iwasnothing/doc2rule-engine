"""
Execute CEL rules with routing logic using mock data.

Usage:
    python 05_execute_rules.py <rules_cel.json> <mockdata.json> [--starting-order <starting_order.json>]

The script will:
  1. Load enriched rules (output of 03_generate_cel.py) and build a
     rule repository keyed by rule_id.
  2. Load mock data (output of 04_generate_data.py) keyed by rule_id.
  3. Optionally load the starting-order JSON (output of 02_build_graph.py)
     to know which rules are root entry points.
  4. For each starting rule and each mock-data row, walk the rule graph
     using the "Compute-then-Route" pattern:
       a. Evaluate calculation_cel  -> store result in output_variable.
       b. Evaluate routing_cel      -> jump to the next rule.
  5. Print a detailed execution trace and save results to a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import celpy


# ============================================================
# PART 1: Context helpers
# ============================================================

_COMPLIANCE_NORMALIZE = {
    "compliant": "Compliant",
    "non-compliant": "Non-Compliant",
    "non_compliant": "Non-Compliant",
    "noncompliant": "Non-Compliant",
}


def _normalize_output_value(value: Any) -> Any:
    """Normalize derived output values for consistency.

    1. Strip leading/trailing whitespace from strings.
    2. Normalize common compliance status variations to Title Case.
    """
    if not isinstance(value, str):
        return value
    # Strip whitespace
    value = value.strip()
    # Normalize compliance casing
    lower = value.lower()
    if lower in _COMPLIANCE_NORMALIZE:
        value = _COMPLIANCE_NORMALIZE[lower]
    return value


def update_context(context: Dict[str, Any], var_path: str, value: Any) -> None:
    """Write *value* into a nested dict following the dotted *var_path*.

    For example, ``update_context(ctx, "derived.grades_served", "PK-12")``
    produces ``ctx["derived"]["grades_served"] = "PK-12"``.

    String values are automatically trimmed and compliance statuses
    are normalised to consistent Title Case.
    """
    if value is None:
        return  # Do not update if calculation resulted in null
    value = _normalize_output_value(value)
    parts = var_path.split(".")
    current = context
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def read_context(context: Dict[str, Any], var_path: str) -> Any:
    """Read a value from a nested dict following the dotted *var_path*.

    Returns None if any segment is missing.
    """
    parts = var_path.split(".")
    current = context
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


# ============================================================
# PART 2: Build the rule repository
# ============================================================

def build_rule_repo(rules: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index a list of enriched rule objects by rule_id."""
    repo: Dict[str, Dict[str, Any]] = {}
    for rule in rules:
        rid = rule.get("rule_id")
        if rid:
            repo[rid] = rule
    return repo


# ============================================================
# PART 3: Merge mock data into a flat context
# ============================================================

def merge_mock_data_for_rules(
    mock_data: Dict[str, Any],
    rule_ids: List[str],
    row_index: int = 0,
) -> Dict[str, Any]:
    """Merge mock data rows from multiple rules into a single nested context.

    For each *rule_id* in *rule_ids*, takes the row at *row_index* (if it
    exists) and deep-merges it into the context.  Later rules overwrite
    earlier ones only when there is a key collision.
    """
    merged: Dict[str, Any] = {}
    for rid in rule_ids:
        rule_data = mock_data.get(rid, {})
        rows = rule_data.get("data", [])
        if rows and row_index < len(rows):
            _deep_merge(merged, rows[row_index])
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ============================================================
# PART 4: CEL Evaluation Helpers
# ============================================================

def _cel_to_python(value: Any) -> Any:
    """Convert celpy types back to native Python types."""
    if value is None:
        return None
    if hasattr(value, "to_json"):
        return value.to_json()
    # celpy may return special types; try common conversions
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sanitize_key(key: str) -> str:
    """Convert a dictionary key to a valid CEL identifier.

    Mirrors the same logic the compiler uses (``_sanitize_cel_identifier``
    in ``03_generate_cel.py``):
    - Replace spaces, hyphens, and other non-word characters with ``_``.
    - Collapse consecutive underscores and strip leading/trailing ``_``.
    - Prefix with ``_`` if the key starts with a digit.
    """
    sanitized = re.sub(r"[^\w]", "_", key)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or key          # fallback to original if empty


def _normalize_context_keys(obj: Any) -> Any:
    """Recursively sanitise dictionary keys so they match the CEL-safe
    identifiers produced by the compiler.

    Handles: digit-starting names, spaces, hyphens, and other special
    characters.  Keys that are already valid identifiers pass through
    unchanged.
    """
    if isinstance(obj, dict):
        return {
            _sanitize_key(k): _normalize_context_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize_context_keys(item) for item in obj]
    return obj


def _coerce_numeric_to_float(obj: Any) -> Any:
    """Recursively convert all int values to float in a nested structure.

    The celpy library maps Python ``int`` to ``IntType`` and ``float`` to
    ``DoubleType``.  CEL is strictly typed and cannot mix the two in
    arithmetic or comparisons.  By coercing every numeric value to float,
    celpy always sees ``DoubleType`` which matches double literals like
    ``100.0`` in CEL expressions.

    This is a safety net that prevents ``no matching overload`` errors
    caused by int/double type mismatches.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numeric_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_to_float(item) for item in obj]
    if isinstance(obj, bool):
        return obj  # bool before int — Python bool is subclass of int
    if isinstance(obj, int):
        return float(obj)
    return obj


def _resolve_null_checks(expression: str, context: Dict[str, Any]) -> str:
    """Pre-resolve ``VAR != null`` and ``VAR == null`` checks at the Python
    level, because celpy cannot compare numeric types against null.

    For each ``VAR != null`` or ``VAR == null`` pattern found in the
    expression, looks up the variable in *context*:
    - If the value exists and is not None → ``!= null`` is ``true``,
      ``== null`` is ``false``.
    - If the value is None or missing → ``!= null`` is ``false``,
      ``== null`` is ``true``.

    Returns the expression with null checks replaced by ``true``/``false``.
    """
    def _lookup(var_path: str, ctx: Dict[str, Any]) -> Any:
        """Walk a dotted path like 'SIS.grade_level' in nested dicts."""
        parts = var_path.split(".")
        current = ctx
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    # Match patterns: VAR != null  or  VAR == null
    # VAR is a dotted identifier like SIS.grade_level
    _CEL_RESERVED = {"true", "false", "null", "in", "double", "int",
                      "string", "size", "has", "type"}

    def _replace_null_check(m):
        var = m.group(1)
        if var in _CEL_RESERVED:
            return m.group(0)  # don't touch keywords
        op = m.group(2)   # == or !=
        value = _lookup(var, context)
        is_null = value is None
        if op == "!=":
            return "false" if is_null else "true"
        else:  # ==
            return "true" if is_null else "false"

    # Match dotted vars (SIS.grade_level) and single-segment vars (x)
    var_pat = r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*"

    resolved = re.sub(
        r"\b(" + var_pat + r")\s*(!=|==)\s*null\b",
        _replace_null_check,
        expression,
    )

    # Also handle: null != VAR  and  null == VAR
    def _replace_null_check_rev(m):
        op = m.group(1)
        var = m.group(2)
        if var in _CEL_RESERVED:
            return m.group(0)
        value = _lookup(var, context)
        is_null = value is None
        if op == "!=":
            return "false" if is_null else "true"
        else:
            return "true" if is_null else "false"

    resolved = re.sub(
        r"\bnull\s*(!=|==)\s*\b(" + var_pat + r")\b",
        _replace_null_check_rev,
        resolved,
    )

    return resolved


def _resolve_bool_comparisons(expression: str, context: Dict[str, Any]) -> str:
    """Pre-resolve ``VAR == true`` and ``VAR == false`` comparisons at the
    Python level.

    Python mock-data may contain ``True``/``False`` (bool), ``'true'``/
    ``'false'`` (str), or even ``'Yes'``/``'No'``.  The celpy runtime is
    strictly typed and may fail to compare these against the CEL boolean
    keywords ``true``/``false``.  By resolving the comparison before CEL
    sees it, we guarantee correct results regardless of the Python type.

    Only resolves when the value can be unambiguously interpreted as boolean.
    """
    var_pat = r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*"

    _CEL_RESERVED = {"true", "false", "null", "in", "double", "int",
                      "string", "size", "has", "type"}

    _SENTINEL = object()

    def _lookup(var_path: str, ctx: Dict[str, Any]) -> Any:
        parts = var_path.split(".")
        current = ctx
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, _SENTINEL)
            else:
                return _SENTINEL
        return current

    def _to_bool(value: Any) -> Optional[bool]:
        """Convert a value to bool if it clearly represents a boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.lower().strip()
            if low == "true":
                return True
            if low == "false":
                return False
        return None  # ambiguous — do not resolve

    def _replace(m):
        var = m.group(1)
        if var in _CEL_RESERVED:
            return m.group(0)
        op = m.group(2)        # == or !=
        cel_bool = m.group(3)  # 'true' or 'false'

        value = _lookup(var, context)
        if value is _SENTINEL:
            return m.group(0)  # variable not found — leave as-is

        py_bool = _to_bool(value)
        if py_bool is None:
            return m.group(0)  # can't determine — leave as-is

        checking_true = (cel_bool == "true")
        if op == "==":
            result = py_bool == checking_true
        else:  # !=
            result = py_bool != checking_true
        return "true" if result else "false"

    resolved = re.sub(
        r"\b(" + var_pat + r")\s*(==|!=)\s*(true|false)\b",
        _replace,
        expression,
    )

    # Also handle reversed form: true == VAR, false != VAR
    def _replace_rev(m):
        cel_bool = m.group(1)
        op = m.group(2)
        var = m.group(3)
        if var in _CEL_RESERVED:
            return m.group(0)
        value = _lookup(var, context)
        if value is _SENTINEL:
            return m.group(0)
        py_bool = _to_bool(value)
        if py_bool is None:
            return m.group(0)
        checking_true = (cel_bool == "true")
        if op == "==":
            result = py_bool == checking_true
        else:
            result = py_bool != checking_true
        return "true" if result else "false"

    resolved = re.sub(
        r"\b(true|false)\s*(==|!=)\s*(" + var_pat + r")\b",
        _replace_rev,
        resolved,
    )

    return resolved


def _fix_string_range_comparisons(expression: str) -> str:
    """Fix string range comparisons that fail lexicographically.

    Converts patterns like ``VAR >= '1' && VAR <= '12'`` to
    ``VAR in ['1', '2', ..., '12']`` because string comparison of
    numeric values uses lexicographic ordering (``'9' > '12'``).
    """
    # Pattern: VAR >= 'N' && VAR <= 'M'
    pattern = (
        r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*>=\s*'(\d+)'\s*&&\s*"
        r"\1\s*<=\s*'(\d+)'"
    )

    def _expand_range(m):
        var = m.group(1)
        low = int(m.group(2))
        high = int(m.group(3))
        if high - low > 50:  # safety: don't expand huge ranges
            return m.group(0)
        items = ", ".join(f"'{i}'" for i in range(low, high + 1))
        return f"{var} in [{items}]"

    fixed = re.sub(pattern, _expand_range, expression)

    # Also handle reversed order: VAR <= 'M' && VAR >= 'N'
    pattern_rev = (
        r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*<=\s*'(\d+)'\s*&&\s*"
        r"\1\s*>=\s*'(\d+)'"
    )

    def _expand_range_rev(m):
        var = m.group(1)
        high = int(m.group(2))
        low = int(m.group(3))
        if high - low > 50:
            return m.group(0)
        items = ", ".join(f"'{i}'" for i in range(low, high + 1))
        return f"{var} in [{items}]"

    fixed = re.sub(pattern_rev, _expand_range_rev, fixed)
    return fixed


def _fix_int_literals_to_double(expression: str) -> str:
    """Convert bare integer literals in CEL expressions to double literals.

    After numeric coercion all data values are ``DoubleType``. CEL integer
    literals are ``IntType``, and celpy cannot compare/operate across the
    two types.  Appending ``.0`` to integer literals makes them
    ``DoubleType``, matching the coerced data.

    Conversions:
    - ``== 2019``  →  ``== 2019.0``
    - ``!= 0``     →  ``!= 0.0``
    - ``? 4 :``    →  ``? 4.0 :``
    - ``[2020, 2022]`` → ``[2020.0, 2022.0]`` (numeric lists only)
    """
    if not expression:
        return expression

    fixed = expression

    # 1. Integer literals next to comparison operators
    def _cmp_int_to_double(m):
        op = m.group(1)
        num = m.group(2)
        after = m.group(3) or ""
        if after.startswith("."):
            return m.group(0)  # already a double
        return f"{op} {num}.0{after}"

    fixed = re.sub(
        r"([><=!]=?)\s*(\d+)(\s|[)\]&|?:,])",
        _cmp_int_to_double,
        fixed,
    )
    fixed = re.sub(
        r"([><=!]=?)\s*(\d+)$",
        lambda m: f"{m.group(1)} {m.group(2)}.0",
        fixed,
    )

    # 2. Integer results in ternary branches: ? N : M  →  ? N.0 : M.0
    fixed = re.sub(
        r"(\?)\s*(\d+)(?!\.\d)\s*(:)",
        lambda m: f"{m.group(1)} {m.group(2)}.0 {m.group(3)}",
        fixed,
    )
    fixed = re.sub(
        r"(:)\s*(\d+)(?!\.\d)\s*([)\s]|$)",
        lambda m: f"{m.group(1)} {m.group(2)}.0 {m.group(3)}",
        fixed,
    )

    # 3. Integer lists → double lists (only if no string elements)
    def _fix_int_list(m):
        content = m.group(1)
        if "'" in content or '"' in content:
            return m.group(0)
        return "[" + re.sub(r"\b(\d+)\b(?!\.)", r"\1.0", content) + "]"

    fixed = re.sub(r"\[([^\]]+)\]", _fix_int_list, fixed)

    # 4. Modulo (%) only works on IntType in celpy.
    #    Convert: `double(VAR) % 2.0) == 0.0` → `int(VAR) % 2) == 0`
    #    Must also fix the comparison after modulo to use int.
    #    Handle the full pattern: (expr % N.0) == M.0
    fixed = re.sub(
        r"\bdouble\(([^)]+)\)\s*%\s*(\d+)(?:\.0)?\s*\)\s*==\s*(\d+)\.0",
        r"int(\1) % \2) == \3",
        fixed,
    )
    fixed = re.sub(
        r"\bdouble\(([^)]+)\)\s*%\s*(\d+)(?:\.0)?\s*\)\s*!=\s*(\d+)\.0",
        r"int(\1) % \2) != \3",
        fixed,
    )
    # Simpler cases without comparison
    fixed = re.sub(
        r"\bdouble\(([^)]+)\)\s*%\s*(\d+)\.0\b",
        r"int(\1) % \2",
        fixed,
    )
    fixed = re.sub(
        r"\bdouble\(([^)]+)\)\s*%\s*(\d+)\b(?!\.)",
        r"int(\1) % \2",
        fixed,
    )

    return fixed


def _fix_size_comparisons(expression: str) -> str:
    """Ensure ``.size()`` comparisons use integer literals, not doubles.

    The CEL ``.size()`` method returns ``IntType``.  If
    ``_fix_int_literals_to_double`` converted ``> 0`` to ``> 0.0`` after a
    ``.size()`` call, this function reverts it.  Also handles ``>=``, ``<``,
    ``<=``, ``==``, ``!=``.
    """
    if not expression or ".size()" not in expression:
        return expression
    return re.sub(
        r"\.size\(\)\s*([><=!]=?)\s*(\d+)\.0\b",
        lambda m: f".size() {m.group(1)} {m.group(2)}",
        expression,
    )


def _rewrite_size_gt_zero_as_nonempty(expression: str, context: Dict[str, Any]) -> str:
    """Rewrite ``VAR.size() > 0`` to ``VAR != ''`` when *VAR* is a string.

    The ``.size()`` method on strings returns the character count as
    ``IntType``.  Comparing ``IntType > IntType`` works, but if the
    condition fails and produces a ``CELEvalError``, the enclosing ternary
    crashes with an overload error.  Rewriting to ``!= ''`` avoids the
    issue entirely.
    """
    if not expression or ".size()" not in expression:
        return expression

    def _lookup(var_path: str, ctx: Dict[str, Any]) -> Any:
        parts = var_path.split(".")
        current = ctx
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _replace(m: re.Match) -> str:
        var = m.group(1)
        op = m.group(2)
        val = m.group(3)
        value = _lookup(var, context)
        if isinstance(value, str):
            # String .size() > 0 → string is non-empty
            if op in (">", ">=") and int(val) == 0:
                return f"{var} != ''"
            if op == "==" and int(val) == 0:
                return f"{var} == ''"
        return m.group(0)

    return re.sub(
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\.size\(\)\s*([><=!]=?)\s*(\d+)",
        _replace,
        expression,
    )


def evaluate_cel_expression(
    env: celpy.Environment,
    expression: str,
    context: Dict[str, Any],
) -> Any:
    """Compile and evaluate a CEL expression against the given context.

    Pre-processing steps:
    1. Resolve ``!= null`` / ``== null`` checks at the Python level
       (celpy can't compare numeric types against null).
    2. Convert integer literals to double literals in the CEL expression
       (celpy can't mix IntType and DoubleType).
    3. Coerce all int data values to float to prevent int/double mixing.
    4. Fix ``.size()`` comparisons that were incorrectly promoted to double.
    5. Rewrite ``.size() > 0`` to ``!= ''`` for string variables.

    Returns the native Python result.
    """
    normalised = _normalize_context_keys(context)

    # Step 1: resolve null checks before CEL sees them
    resolved_expr = _resolve_null_checks(expression, normalised)

    # Step 2: resolve bool comparisons (Python bool/str vs CEL true/false)
    resolved_expr = _resolve_bool_comparisons(resolved_expr, normalised)

    # Step 3: fix string range comparisons (lexicographic → list membership)
    resolved_expr = _fix_string_range_comparisons(resolved_expr)

    # Step 4: convert integer literals to double in the expression
    resolved_expr = _fix_int_literals_to_double(resolved_expr)

    # Step 5: revert .size() comparisons back to int (they get wrongly
    #         promoted to double by Step 4)
    resolved_expr = _fix_size_comparisons(resolved_expr)

    # Step 6: rewrite .size() > 0 to != '' for string values in context
    resolved_expr = _rewrite_size_gt_zero_as_nonempty(resolved_expr, normalised)

    # Step 7: compile and evaluate with coerced numeric types
    ast = env.compile(resolved_expr)
    prg = env.program(ast)
    coerced = _coerce_numeric_to_float(normalised)
    activation = celpy.json_to_cel(coerced)
    result = prg.evaluate(activation)
    py_result = _cel_to_python(result)

    # Step 8: handle infinity from unguarded division by zero
    if isinstance(py_result, float) and (
        py_result == float("inf") or py_result == float("-inf")
    ):
        return None

    return py_result


# ============================================================
# PART 5: The Compute-then-Route Engine
# ============================================================

MAX_STEPS = 200  # safety limit to prevent infinite loops
LOOP_DETECTION_WINDOW = 6  # detect repeating rule patterns within this window


def _detect_runtime_loop(visited_ids: List[str], window: int = LOOP_DETECTION_WINDOW) -> bool:
    """Detect if the last *window* rule visits form a repeating pattern.

    Catches:
    - Self-loops:     [..., A, A, A]
    - Ping-pong:      [..., A, B, A, B, A, B]
    - 3-node cycles:  [..., A, B, C, A, B, C]

    Returns True if a loop is detected.
    """
    if len(visited_ids) < window:
        return False
    recent = visited_ids[-window:]
    # Check for repeating cycles of length 1, 2, or 3
    for cycle_len in (1, 2, 3):
        if len(recent) >= cycle_len * 2:
            tail = recent[-cycle_len * 2:]
            pattern = tail[:cycle_len]
            if all(tail[i] == pattern[i % cycle_len] for i in range(len(tail))):
                return True
    return False


def execute_rule_engine(
    rule_repo: Dict[str, Dict[str, Any]],
    start_id: str,
    initial_data: Dict[str, Any],
    mock_data: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    row_index: int = 0,
) -> Dict[str, Any]:
    """Walk the rule graph from *start_id* using the Compute-then-Route pattern.

    Parameters
    ----------
    rule_repo : dict
        Rule objects keyed by rule_id.
    start_id : str
        The rule_id to begin execution.
    initial_data : dict
        Initial context (nested dict) to seed the engine.
    mock_data : dict, optional
        Full mock-data payload keyed by rule_id.  When the engine arrives
        at a rule, any missing input variables are injected from the
        rule's mock-data row so that calculation_cel can succeed even
        when the initial context doesn't contain every variable.
    verbose : bool
        Print a step-by-step trace.
    row_index : int
        Which mock-data row to use when injecting additional data for
        chained rules (default 0).

    Returns
    -------
    dict
        ``{ "context": <final context>, "trace": [<step dicts>] }``
    """
    import copy
    env = celpy.Environment()
    context = copy.deepcopy(initial_data)
    current_id: Optional[str] = start_id
    trace: List[Dict[str, Any]] = []
    visited_ids: List[str] = []
    steps = 0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  STARTING ENGINE at {start_id}")
        print(f"{'─'*60}")

    while current_id:
        if steps >= MAX_STEPS:
            if verbose:
                print(f"  ⚠️  SAFETY LIMIT ({MAX_STEPS} steps) reached — stopping.")
            break

        # Early loop detection — break out as soon as a pattern repeats
        visited_ids.append(current_id)
        if steps >= LOOP_DETECTION_WINDOW and _detect_runtime_loop(visited_ids):
            if verbose:
                recent = visited_ids[-LOOP_DETECTION_WINDOW:]
                print(f"  ⚠️  LOOP DETECTED (pattern: {' -> '.join(recent)}) — stopping.")
            break

        steps += 1

        rule = rule_repo.get(current_id)
        if not rule:
            if verbose:
                print(f"  ✅ END: Reached unknown or terminal state '{current_id}'")
            trace.append({"step": steps, "rule_id": current_id, "status": "unknown_or_terminal"})
            break

        # Skip rules flagged as manual_review or non-computational
        if rule.get("manual_review"):
            skip_reason = rule.get("skip_reason", "manual_review")
            if verbose:
                print(f"  ⏭️  Skipping {current_id} ({skip_reason}) — no executable CEL")
            trace.append({
                "step": steps,
                "rule_id": current_id,
                "rule_name": rule.get("rule_name", ""),
                "status": f"skipped_{skip_reason}",
            })
            current_id = None
            break

        rule_name = rule.get("rule_name", "")
        if verbose:
            print(f"\n  ► Step {steps}: {current_id} — {rule_name}")

        step_record: Dict[str, Any] = {
            "step": steps,
            "rule_id": current_id,
            "rule_name": rule_name,
            "context_updates": [],
        }

        # --- Inject additional mock data for this rule if available ---
        # For the starting rule the initial_data already contains the
        # correct row; for chained rules we inject the row at row_index
        # (falling back to row 0 if the index is out of range).
        if mock_data and current_id in mock_data and current_id != start_id:
            rows = mock_data[current_id].get("data", [])
            if rows:
                idx = row_index if row_index < len(rows) else 0
                _deep_merge(context, rows[idx])

        # --- STEP 1: CALCULATE & UPDATE STATE ---
        calc_cel = rule.get("calculation_cel")
        output_var = rule.get("output_variable")

        step_record["calculation_cel"] = calc_cel

        if calc_cel:
            try:
                calculated_value = evaluate_cel_expression(env, calc_cel, context)
            except ZeroDivisionError:
                calculated_value = None
                if verbose:
                    print(f"    ℹ️  Division by zero — returning None")
            except celpy.CELEvalError as e:
                # celpy wraps ZeroDivisionError inside CELEvalError
                if "divide by zero" in str(e).lower() or "ZeroDivisionError" in str(e):
                    calculated_value = None
                    if verbose:
                        print(f"    ℹ️  Division by zero (CEL-wrapped) — returning None")
                else:
                    calculated_value = f"ERROR: {e}"
                    if verbose:
                        print(f"    ⚠️  CEL runtime error: {e}")
            except celpy.CELParseError as e:
                calculated_value = f"ERROR: {e}"
                if verbose:
                    print(f"    ⚠️  CEL compile error: {e}")
            except Exception as e:
                # Also catch wrapped ZeroDivisionError from other exception types
                if "divide by zero" in str(e).lower() or "ZeroDivisionError" in str(e):
                    calculated_value = None
                    if verbose:
                        print(f"    ℹ️  Division by zero — returning None")
                else:
                    calculated_value = f"ERROR: {e}"
                    if verbose:
                        print(f"    ⚠️  Calculation error: {e}")

            if output_var:
                update_context(context, output_var, calculated_value)
                step_record["context_updates"].append({
                    "variable": output_var,
                    "value": _safe_serialize(calculated_value),
                })
                if verbose:
                    print(f"    💾 State Updated: {output_var} = {calculated_value}")

            step_record["output_variable"] = output_var
            step_record["calculated_value"] = _safe_serialize(calculated_value)
        else:
            if verbose:
                print(f"    (no calculation_cel)")
            step_record["calculated_value"] = None

        # --- STEP 2: ROUTE ---
        routing_cel = rule.get("routing_cel")
        step_record["routing_cel"] = routing_cel

        if routing_cel:
            try:
                # Re-evaluate with updated context (includes derived values)
                next_id = evaluate_cel_expression(env, routing_cel, context)
            except Exception as e:
                next_id = None
                if verbose:
                    print(f"    ⚠️  Routing error: {e}")

            if verbose:
                print(f"    ↪ Routing Decision: → {next_id}")

            step_record["next_rule"] = next_id
            current_id = next_id
        else:
            if verbose:
                print(f"    No routing logic — stopping.")
            step_record["next_rule"] = None
            current_id = None

        # --- Snapshot context after this step ---
        step_record["context_snapshot"] = _snapshot_context(context)

        trace.append(step_record)

    return {"context": context, "trace": trace}


def _safe_serialize(value: Any) -> Any:
    """Ensure a value is JSON-serialisable (fallback to str)."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def _snapshot_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Create a JSON-safe deep copy of the context for logging."""
    try:
        return json.loads(json.dumps(context, default=str))
    except (TypeError, ValueError):
        return {"_error": "context could not be serialised"}


def _flatten_context(context: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested context dict into dotted-path keys.

    ``{"derived": {"grades_served": "PK-12"}}``
    becomes ``{"derived.grades_served": "PK-12"}``.
    """
    flat: Dict[str, Any] = {}
    for key, value in context.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_context(value, full_key))
        else:
            flat[full_key] = _safe_serialize(value)
    return flat


# ============================================================
# PART 6: Execution report
# ============================================================

def run_all_starting_rules(
    rule_repo: Dict[str, Dict[str, Any]],
    mock_data: Dict[str, Any],
    starting_rules: List[str],
    num_rows: int = 1,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Execute the engine for every starting rule and every mock-data row.

    Returns a list of result dicts suitable for JSON serialisation.
    """
    results: List[Dict[str, Any]] = []

    for start_id in starting_rules:
        if start_id not in rule_repo:
            if verbose:
                print(f"\n⚠️  Starting rule '{start_id}' not found in rule repo — skipping.")
            continue

        # Determine how many rows are available for this starting rule
        rule_mock = mock_data.get(start_id, {})
        available_rows = len(rule_mock.get("data", []))
        rows_to_run = min(num_rows, max(available_rows, 1))

        for row_idx in range(rows_to_run):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  START RULE: {start_id}  |  DATA ROW: {row_idx}")
                print(f"{'='*60}")

            # Build initial context from this rule's mock data row
            initial_data = merge_mock_data_for_rules(
                mock_data, [start_id], row_index=row_idx,
            )

            result = execute_rule_engine(
                rule_repo=rule_repo,
                start_id=start_id,
                initial_data=initial_data,
                mock_data=mock_data,
                verbose=verbose,
                row_index=row_idx,
            )

            # Build a flat variable summary for the final context
            final_ctx = result["context"]
            context_variables = _flatten_context(final_ctx)

            if verbose:
                print(f"\n  {'─'*50}")
                print(f"  CONTEXT VARIABLES after all rules:")
                for var_path, var_val in sorted(context_variables.items()):
                    print(f"    {var_path} = {var_val}")
                print(f"  {'─'*50}")

            results.append({
                "start_rule": start_id,
                "data_row_index": row_idx,
                "initial_data": initial_data,
                "final_context": final_ctx,
                "context_variables": context_variables,
                "trace": result["trace"],
                "total_steps": len(result["trace"]),
            })

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a compact summary table of all execution runs."""
    print(f"\n{'='*70}")
    print("  EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Start Rule':<14} {'Row':<5} {'Steps':<7} {'Path'}")
    print(f"  {'-'*14} {'-'*5} {'-'*7} {'-'*40}")

    for r in results:
        path_ids = [s["rule_id"] for s in r["trace"]]
        path_str = " → ".join(path_ids)
        if len(path_str) > 60:
            path_str = path_str[:57] + "..."
        print(
            f"  {r['start_rule']:<14} {r['data_row_index']:<5} "
            f"{r['total_steps']:<7} {path_str}"
        )

    total_runs = len(results)
    total_steps = sum(r["total_steps"] for r in results)
    print(f"\n  Total runs: {total_runs}  |  Total steps: {total_steps}")
    print(f"{'='*70}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute CEL rules with routing using mock data.",
    )
    parser.add_argument(
        "rules_file",
        help="Path to the enriched rules JSON (e.g., *_cel.json).",
    )
    parser.add_argument(
        "mockdata_file",
        help="Path to the mock-data JSON (e.g., *_mockdata.json).",
    )
    parser.add_argument(
        "--starting-order",
        default=None,
        help=(
            "Path to the starting-order JSON (e.g., *_starting_order.json). "
            "If omitted, the script auto-detects root rules (those never "
            "referenced as a routing target)."
        ),
    )
    parser.add_argument(
        "--num-rows", "-n",
        type=int,
        default=1,
        help="Number of mock-data rows to execute per starting rule (default: 1).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress step-by-step trace output.",
    )
    args = parser.parse_args()

    # ── Validate input files ──────────────────────────────────────────────
    for fpath in [args.rules_file, args.mockdata_file]:
        if not os.path.isfile(fpath):
            print(f"Error: File not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    # ── 1. Load rules and build repo ──────────────────────────────────────
    with open(args.rules_file, "r", encoding="utf-8") as f:
        rules_list: list[dict] = json.load(f)

    rule_repo = build_rule_repo(rules_list)
    print(f"Loaded {len(rule_repo)} rules from: {args.rules_file}")

    # ── 2. Load mock data ─────────────────────────────────────────────────
    with open(args.mockdata_file, "r", encoding="utf-8") as f:
        mock_data: Dict[str, Any] = json.load(f)

    total_rows = sum(len(v.get("data", [])) for v in mock_data.values())
    print(f"Loaded mock data for {len(mock_data)} rules ({total_rows} total rows) from: {args.mockdata_file}")

    # ── 3. Determine starting rules ───────────────────────────────────────
    if args.starting_order and os.path.isfile(args.starting_order):
        with open(args.starting_order, "r", encoding="utf-8") as f:
            starting_rules: list[str] = json.load(f)
        print(f"Loaded {len(starting_rules)} starting rules from: {args.starting_order}")
    else:
        # Auto-detect: rules that are never a routing target
        all_targets: set[str] = set()
        for rule in rules_list:
            for route in rule.get("outgoing_routes", []):
                target = route.get("next_rule")
                if target:
                    all_targets.add(target)
        starting_rules = sorted(rid for rid in rule_repo if rid not in all_targets)
        print(f"Auto-detected {len(starting_rules)} starting rules (root nodes).")

    # ── 4. Execute ────────────────────────────────────────────────────────
    verbose = not args.quiet
    results = run_all_starting_rules(
        rule_repo=rule_repo,
        mock_data=mock_data,
        starting_rules=starting_rules,
        num_rows=args.num_rows,
        verbose=verbose,
    )

    # ── 5. Summary ────────────────────────────────────────────────────────
    print_summary(results)

    # ── 6. Save execution report ──────────────────────────────────────────
    base, ext = os.path.splitext(args.rules_file)
    # Strip trailing _cel or _jsonlogic if present for cleaner naming
    for suffix in ("_cel", "_jsonlogic"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{base}_execution_{timestamp}.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "rules_file": args.rules_file,
        "mockdata_file": args.mockdata_file,
        "starting_rules_count": len(starting_rules),
        "total_runs": len(results),
        "total_steps": sum(r["total_steps"] for r in results),
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nExecution report saved to: {output_file}")
