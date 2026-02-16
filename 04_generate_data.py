"""
Generate dynamic mock data for CEL rule execution.

Usage:
    python 04_generate_data.py <rules_cel.json> [--num-rows 5]

The script will:
  1. Read the enriched rules JSON (output of 03_generate_cel.py).
  2. For each rule, extract all input variables from calculation_cel by
     parsing CEL identifier references (e.g. SIS.grade_level).
  3. Cross-reference with data_required for context (descriptions, examples).
  4. Use an LLM to infer the best Faker provider + kwargs for each variable.
  5. Generate mock data rows using Faker.
  6. Save as <input_stem>_mockdata.json — nested dicts ready to feed into
     CEL evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from faker import Faker
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

fake = Faker()
Faker.seed(0)
random.seed(0)

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


# ==========================================
# Pydantic Models
# ==========================================

class VariableSpec(BaseModel):
    """Specification for how to generate a single input variable."""
    variable_path: str = Field(
        ...,
        description="The exact variable path as used in the CEL expression (e.g., 'SIS.grade_level').",
    )
    description: str = Field(
        ...,
        description="Brief description of what this variable represents.",
    )
    data_type: str = Field(
        ...,
        description="Python type: str, int, float, bool.",
    )
    faker_provider: str = Field(
        ...,
        description=(
            "The Faker provider method to call "
            "(e.g., 'random_element', 'random_int', 'boolean', 'bothify')."
        ),
    )
    faker_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Keyword arguments to pass to the Faker provider "
            "(e.g., {'elements': ['A', 'B']} for random_element)."
        ),
    )


class VariableSchemaResult(BaseModel):
    """LLM output: a list of generation specs for every input variable."""
    variables: List[VariableSpec] = Field(
        ...,
        description="List of all input variables with their generation specs.",
    )


# ==========================================
# PART 1: Extract variables from CEL expressions
# ==========================================

# Regex to find dotted identifiers in CEL (e.g. SIS.grade_level, EPS.school_id).
# Matches: word_char_sequence DOT word_char_sequence (one or more dots).
# Excludes standalone words (no dot) to avoid matching CEL keywords/literals.
_CEL_DOTTED_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b")

# CEL built-in functions / keywords that should NOT be treated as variables.
_CEL_BUILTINS = {
    "double", "int", "uint", "string", "bool", "bytes", "list", "map",
    "null", "true", "false", "size", "contains", "startsWith", "endsWith",
    "matches", "type", "has", "all", "exists", "exists_one", "filter",
    "timestamp", "duration",
}

# CEL method names that can appear as the last segment of a dotted path
# (e.g. EPS.school_code.startsWith  →  method call, not a field).
_CEL_METHODS = {
    "contains", "startsWith", "endsWith", "matches", "size",
    "exists", "exists_one", "all", "filter", "map",
}


def extract_variables_from_cel(expression: str) -> set[str]:
    """Extract all dotted variable references from a CEL expression string.

    For example, from:
        ``SIS.grade_level >= 12 && EPS.school_code.startsWith('3')``
    returns:
        ``{"SIS.grade_level", "EPS.school_code"}``

    - Single-quoted string literals are stripped before scanning so that
      dotted substrings inside strings (e.g. ``'foo.bar'``) are ignored.
    - CEL method calls on the tail (e.g. ``.startsWith``, ``.contains``)
      are trimmed so the actual variable path is returned.
    """
    # Remove single-quoted string literals to avoid false positives
    cleaned = re.sub(r"'(?:[^'\\]|\\.)*'", "", expression)
    matches = _CEL_DOTTED_IDENT_RE.findall(cleaned)
    variables: set[str] = set()
    for m in matches:
        segments = m.split(".")
        # Skip if the first segment is a builtin (e.g. "string.format")
        if segments[0].lower() in _CEL_BUILTINS:
            continue
        # Strip trailing CEL method names (e.g. EPS.school_code.startsWith)
        while len(segments) > 1 and segments[-1] in _CEL_METHODS:
            segments = segments[:-1]
        # After stripping, need at least a two-part dotted path (Source.field)
        if len(segments) >= 2:
            variables.add(".".join(segments))
    return variables


def gather_variable_context(
    rule: Dict[str, Any],
    var_paths: set[str],
) -> List[Dict[str, Any]]:
    """Cross-reference variable paths with data_required to gather context.

    Builds a lookup from the rule's ``data_required`` schema and matches
    each variable path by its leaf name (the part after the last dot).
    Returns a list of dicts carrying whatever context is available
    (description, example_values, data_type).
    """
    # Build lookup: attribute_name -> attribute dict
    attr_lookup: Dict[str, Dict[str, Any]] = {}
    for source in rule.get("data_required", []):
        for attr in source.get("data_attributes", []):
            a_name = attr.get("attribute_name", "")
            if a_name:
                attr_lookup[a_name] = attr

    result: List[Dict[str, Any]] = []
    for var_path in sorted(var_paths):
        context: Dict[str, Any] = {"variable_path": var_path}

        # Match by leaf name (e.g., "SIS.grade_level" -> "grade_level")
        leaf = var_path.rsplit(".", 1)[-1] if "." in var_path else var_path
        matched = attr_lookup.get(leaf) or attr_lookup.get(var_path)

        if matched:
            context["description"] = matched.get("description", "")
            context["example_values"] = matched.get("example_values", [])
            context["data_type"] = matched.get("data_type", "string")

        result.append(context)

    return result


# ==========================================
# PART 2: LLM Schema Inference
# ==========================================

def infer_variable_specs(
    rule: Dict[str, Any],
    variable_contexts: List[Dict[str, Any]],
) -> List[VariableSpec]:
    """Use an LLM to determine the best Faker provider for each variable.

    The LLM receives the calculation_cel (to see which literal values
    each variable is compared against) plus any available context from
    data_required (descriptions, example_values, data_type).
    """
    llm = _get_llm()

    system_prompt = """\
### SYSTEM ROLE
You are a QA Data Engineer. Your task is to analyze a business rule's \
CEL (Common Expression Language) expressions and determine how to generate \
realistic mock data for each input variable using Python's Faker library.

### INPUT
You will receive:
1. A business rule with its `calculation_cel` (a CEL expression) and metadata.
2. A list of input variables extracted from the expression, with any available \
context (description, example_values, data_type from the rule schema).

### YOUR TASK
For EACH input variable return:
1. `variable_path` — the **exact** dotted path as used in the CEL expression \
(e.g. `SIS.grade_level`) — do NOT rename.
2. `description` — what this variable represents.
3. `data_type` — one of: str, int, float, bool.
4. `faker_provider` — a **real** Faker method name. Prefer these:
   - Picking from a known set → `random_element`
   - Random integers → `random_int`
   - Random floats → `pyfloat`
   - Booleans → `boolean`
   - Human names → `name`
   - IDs / codes with a pattern → `bothify` (e.g. text="SCH########")
   - Dates → `date`
   - Generic strings → `word` or `sentence`
5. `faker_kwargs` — keyword arguments for the provider.

### CRITICAL RULES — READ CAREFULLY

#### RULE 1: TYPE MATCHING (MOST IMPORTANT)
You MUST match the **exact type and format** that the CEL expression uses \
for comparisons. Inspect every `==`, `!=`, `in`, `>=`, `<=` operator:
- If the CEL compares to a **string** like `'true'` or `'false'`, \
generate STRING values `"true"` / `"false"` — NOT boolean `true`/`false`.
- If the CEL compares to **string numbers** like `'4'` or `'8'`, \
generate STRING values `"4"`, `"8"` — NOT integers 4, 8.
- If the CEL compares to **integers** like `10` (no quotes), \
generate INTEGER values — set `data_type: "int"`.
- If the CEL uses `>=` or `<=` with numbers, the variable must be \
numeric (int or float), not a string.

#### RULE 2: BRANCH COVERAGE (INCLUSION CHECKS)
If the CEL uses `X in ['A', 'B']` or `X == 'C'`, you MUST use \
`faker_provider: "random_element"` and put **ALL** of those literal \
values into `faker_kwargs.elements`, plus ONE extra plausible value \
to test the else/null branch.

#### RULE 3: EXCLUSION CHECKS — DO THE OPPOSITE
If the CEL EXCLUDES certain values (e.g., \
`!(X in ['01','02','03'])` meaning "X must NOT be in ['01','02','03']"), \
then your elements list should contain MOSTLY values that are **NOT** \
in the exclusion list, plus one or two values from the exclusion list \
for edge-case testing.

#### RULE 4: GATEKEEPER CONDITIONS (CRITICAL)
Many rules have a gatekeeper ternary at the top level that returns \
`'Not Applicable'` or a meaningful else value when the condition fails. \
You MUST ensure the generated data satisfies the gatekeeper condition \
**at least 60% of the time** so that the positive/true branch is exercised. \
For example, if the gate is `X == 'active' ? (...) : 'Not Applicable'`, \
then your elements list should have `"active"` appearing frequently \
(weight it by putting it multiple times or making it 60-70% of elements). \
This is CRITICAL for branch coverage — every rule MUST have at least one \
data row that triggers its positive path.

#### RULE 5: DATE AND RANGE VALUES
If the CEL compares dates (e.g., `>= '2023-10-01'`), generate \
dates in the relevant range using `random_element` with specific \
date strings that both pass and fail the comparison.
If the CEL compares numeric ranges (e.g., `>= 10`), use \
`random_int` with min/max that straddles the threshold \
(e.g., min=5, max=20 for a >= 10 check).

#### RULE 6: OTHER RULES
- For `random_int` always supply `{{"min": ..., "max": ...}}`.
- For `bothify` always supply `{{"text": "PATTERN"}}`.
- Do NOT include variables starting with "derived." — those are outputs.
- If `example_values` are provided, incorporate them but ONLY if they \
match the comparison type in the CEL expression."""

    rule_summary = {
        "rule_id": rule.get("rule_id"),
        "rule_name": rule.get("rule_name"),
        "calculation_cel": rule.get("calculation_cel"),
        "routing_cel": rule.get("routing_cel"),
        "conditions": rule.get("conditions", []),
    }

    user_input = json.dumps(
        {
            "rule_context": rule_summary,
            "variables_to_generate": variable_contexts,
        },
        indent=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = prompt | llm.with_structured_output(VariableSchemaResult)

    print("  🧐 LLM: Inferring variable types and Faker providers...")
    result = chain.invoke({"input": user_input})
    return result.variables


# ==========================================
# PART 2b: Post-LLM Validation — fix type mismatches
# ==========================================

# Regex patterns for detecting literal comparisons in CEL:
#   VAR == 'string'  /  VAR != 'string'  /  VAR in ['a', 'b']
_STRING_CMP_RE = re.compile(
    r"""(?:                           # match one of:
        \b(\w+(?:\.\w+)+)\s*[!=]=\s*'  # VAR == 'x'  or  VAR != 'x'
      | '\w*'\s*[!=]=\s*(\w+(?:\.\w+)+) # 'x' == VAR  (reversed)
    )""",
    re.VERBOSE,
)

# VAR compared to a bare integer:  VAR >= 10, VAR == 100
_INT_CMP_RE = re.compile(
    r"\b(\w+(?:\.\w+)+)\s*(?:[><=!]=?)\s*(\d+)\b"
)


def validate_specs_against_cel(
    specs: List[VariableSpec],
    cel_expression: str,
    routing_cel: Optional[str] = None,
) -> List[VariableSpec]:
    """Post-LLM validation: override ``data_type`` when the CEL expression
    makes the expected type unambiguous.

    The most common LLM mistake is setting ``data_type: "bool"`` for a
    variable the CEL compares to the *string* ``'true'`` / ``'false'``.
    This function detects such mismatches and forces the correct type.
    """
    combined = cel_expression or ""
    if routing_cel:
        combined += " " + routing_cel

    # Build a set of variable paths that are compared to string literals
    string_compared_vars: set[str] = set()
    for m in _STRING_CMP_RE.finditer(combined):
        var = m.group(1) or m.group(2)
        if var:
            string_compared_vars.add(var)

    # Also catch `VAR in ['a', 'b', ...]`
    in_list_re = re.compile(r"\b(\w+(?:\.\w+)+)\s+in\s+\[")
    for m in in_list_re.finditer(combined):
        # Check if the list contains string literals (single-quoted)
        start = m.end()
        bracket_end = combined.find("]", start)
        if bracket_end > start:
            list_content = combined[start:bracket_end]
            if "'" in list_content:
                string_compared_vars.add(m.group(1))

    # Build a set of variable paths compared to bare integers
    int_compared_vars: set[str] = set()
    for m in _INT_CMP_RE.finditer(combined):
        var = m.group(1)
        # Only if the literal is NOT inside a single-quoted string
        pos = m.start()
        preceding = combined[:pos]
        # Count unmatched single quotes — if odd, we're inside a string
        if preceding.count("'") % 2 == 0:
            int_compared_vars.add(var)

    # Detect variables used as denominators in division:
    #   ... / VAR  or  ... / double(VAR)  or  ... / (VAR ... ? 1 : 0)
    denominator_vars: set[str] = set()
    # Pattern: / followed by a dotted variable (possibly wrapped in double())
    denom_re = re.compile(
        r"/\s*(?:double\s*\(\s*)?(\w+(?:\.\w+)+)"
    )
    for m in denom_re.finditer(combined):
        denominator_vars.add(m.group(1))
    # Also catch pattern: / (VAR == 'X' ? 1 : 0) — the var is used to produce
    # the denominator value, so it needs to sometimes be 'X' to avoid zero.
    denom_ternary_re = re.compile(
        r"/\s*\(?(\w+(?:\.\w+)+)\s*[!=]="
    )
    for m in denom_ternary_re.finditer(combined):
        denominator_vars.add(m.group(1))

    fixes_applied = 0
    corrected: List[VariableSpec] = []
    for spec in specs:
        vp = spec.variable_path

        if vp in string_compared_vars and spec.data_type != "str":
            print(f"     🔧 Fix: {vp} data_type '{spec.data_type}' → 'str' "
                  f"(CEL compares to string literal)")
            spec = spec.model_copy(update={"data_type": "str"})
            fixes_applied += 1

        elif vp in int_compared_vars and spec.data_type not in ("int", "float"):
            # Don't override if already in string_compared_vars (string takes priority)
            if vp not in string_compared_vars:
                print(f"     🔧 Fix: {vp} data_type '{spec.data_type}' → 'int' "
                      f"(CEL compares to integer literal)")
                spec = spec.model_copy(update={"data_type": "int"})
                fixes_applied += 1

        # Ensure denominator variables avoid generating zero
        if vp in denominator_vars:
            if spec.data_type in ("int", "float"):
                # For numeric denominators, ensure min >= 1
                kw = dict(spec.faker_kwargs) if spec.faker_kwargs else {}
                if spec.faker_provider == "random_int":
                    if kw.get("min", 0) <= 0:
                        kw["min"] = 1
                        print(f"     🔧 Fix: {vp} random_int min → 1 "
                              f"(used as denominator, avoid div-by-zero)")
                        spec = spec.model_copy(update={"faker_kwargs": kw})
                        fixes_applied += 1
                elif spec.faker_provider == "pyfloat":
                    kw.setdefault("min_value", 0.1)
                    if kw["min_value"] <= 0:
                        kw["min_value"] = 0.1
                    print(f"     🔧 Fix: {vp} pyfloat min_value → {kw['min_value']} "
                          f"(used as denominator, avoid div-by-zero)")
                    spec = spec.model_copy(update={"faker_kwargs": kw})
                    fixes_applied += 1
            elif spec.data_type == "str" and spec.faker_provider == "random_element":
                # For string denominators in patterns like:
                #   / (VAR == 'Y' ? 1 : 0) — ensure the matching value dominates
                kw = dict(spec.faker_kwargs) if spec.faker_kwargs else {}
                elements = kw.get("elements", [])
                if elements:
                    # Find which values appear in the division context as the
                    # "positive" match (the value that makes the denominator 1).
                    # Boost it to ~80% of elements to avoid div-by-zero.
                    # Look for: / (VAR == 'X' ? 1 : 0)
                    pos_re = re.compile(
                        rf"/\s*\(?\s*{re.escape(vp)}\s*==\s*'([^']+)'"
                    )
                    for pm in pos_re.finditer(combined):
                        pos_val = pm.group(1)
                        if pos_val in elements:
                            # Boost: make 80% of elements the positive value
                            total = max(len(elements), 5)
                            pos_count = int(total * 0.8)
                            neg_elements = [e for e in elements if e != pos_val]
                            new_elements = ([pos_val] * pos_count +
                                            neg_elements[:max(total - pos_count, 1)])
                            kw["elements"] = new_elements
                            print(f"     🔧 Fix: {vp} boosted '{pos_val}' to ~80% "
                                  f"(used as denominator, avoid div-by-zero)")
                            spec = spec.model_copy(update={"faker_kwargs": kw})
                            fixes_applied += 1
                            break

        corrected.append(spec)

    if fixes_applied:
        print(f"     📋 Validated specs: {fixes_applied} type override(s) applied")

    return corrected


# ==========================================
# PART 3: Dynamic Data Generation (Faker)
# ==========================================

def generate_dynamic_data(
    specs: List[VariableSpec],
    num_rows: int = 5,
) -> List[Dict[str, Any]]:
    """Generate *num_rows* of mock data by calling Faker providers.

    Each row is a flat dict keyed by variable_path.
    """
    rows: List[Dict[str, Any]] = []
    print(f"  🎲 FAKER: Generating {num_rows} data row(s)...")

    for _ in range(num_rows):
        row: Dict[str, Any] = {}
        for spec in specs:
            row[spec.variable_path] = _invoke_faker(spec)
        rows.append(row)

    return rows


def _invoke_faker(spec: VariableSpec) -> Any:
    """Call a Faker provider dynamically from a VariableSpec."""
    provider_name = spec.faker_provider
    kwargs = dict(spec.faker_kwargs) if spec.faker_kwargs else {}

    if hasattr(fake, provider_name):
        method = getattr(fake, provider_name)
        try:
            value = method(**kwargs)
        except Exception:
            # kwargs incompatible — try without, then fallback
            try:
                value = method()
            except Exception:
                value = _fallback_by_type(spec.data_type)
    else:
        # Provider doesn't exist on Faker — fall back by type
        value = _fallback_by_type(spec.data_type)

    return _coerce(value, spec.data_type)


def _fallback_by_type(data_type: str) -> Any:
    """Generate a sensible default when the Faker provider is unavailable."""
    if data_type == "int":
        return random.randint(0, 1000)
    if data_type == "float":
        return round(random.uniform(0, 100), 2)
    if data_type == "bool":
        return fake.boolean()
    return fake.word()


def _coerce(value: Any, data_type: str) -> Any:
    """Best-effort coercion to the declared Python type."""
    try:
        if data_type == "int":
            return int(value) if not isinstance(value, int) else value
        if data_type == "float":
            return float(value) if not isinstance(value, float) else value
        if data_type == "bool":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes")
        return str(value) if not isinstance(value, str) else value
    except (ValueError, TypeError):
        return value


# ==========================================
# PART 3b: Golden Row — ensures positive-path coverage
# ==========================================

def _build_golden_row(
    cel_expression: str,
    specs: List[VariableSpec],
) -> Optional[Dict[str, Any]]:
    """Build a "golden" data row that satisfies the primary positive condition.

    Analyses the CEL expression to find literal values used in equality
    comparisons and ``in`` checks, then constructs a row where each
    variable is set to the value most likely to trigger the positive
    (non-null, non-else) path.

    Returns ``None`` if no useful golden row can be inferred.
    """
    if not cel_expression or not specs:
        return None

    # Build spec lookup
    spec_by_path: Dict[str, VariableSpec] = {s.variable_path: s for s in specs}

    golden: Dict[str, Any] = {}
    matched = False

    # Pattern 1: VAR == 'literal'
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.\w+)+)\s*==\s*'([^']*)'",
        cel_expression,
    ):
        var, val = m.group(1), m.group(2)
        if var in spec_by_path and var not in golden:
            golden[var] = val
            matched = True

    # Pattern 2: VAR in ['a', 'b', 'c'] — pick the first element
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.\w+)+)\s+in\s+\[([^\]]+)\]",
        cel_expression,
    ):
        var = m.group(1)
        list_content = m.group(2)
        if var in spec_by_path and var not in golden:
            elements = re.findall(r"'([^']*)'", list_content)
            if elements:
                golden[var] = elements[0]
                matched = True

    # Pattern 3: VAR == true (boolean)
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.\w+)+)\s*==\s*(true|false)\b",
        cel_expression,
    ):
        var, val = m.group(1), m.group(2)
        if var in spec_by_path and var not in golden:
            spec = spec_by_path[var]
            if spec.data_type == "bool":
                golden[var] = val == "true"
            else:
                golden[var] = val
            matched = True

    # Pattern 4: double(VAR) >= N — use a value above the threshold
    for m in re.finditer(
        r"double\(\s*([a-zA-Z_]\w*(?:\.\w+)+)\s*\)\s*>=\s*(\d+(?:\.\d+)?)",
        cel_expression,
    ):
        var, threshold = m.group(1), m.group(2)
        if var in spec_by_path and var not in golden:
            golden[var] = float(threshold) + 1.0
            matched = True

    if not matched:
        return None

    # Fill remaining specs with default values from their Faker specs
    for spec in specs:
        if spec.variable_path not in golden:
            golden[spec.variable_path] = _invoke_faker(spec)

    return golden


# ==========================================
# PART 4: Nest flat keys for CEL evaluation
# ==========================================

def nest_flat_row(flat_row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a flat dict with dotted keys into a nested dict.

    CEL's ``SIS.grade_level`` traverses nested objects, so
    ``{"SIS.grade_level": "5"}`` must become
    ``{"SIS": {"grade_level": "5"}}``.
    """
    nested: Dict[str, Any] = {}
    for key, value in flat_row.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mock data for CEL rule execution.",
    )
    parser.add_argument(
        "rules_file",
        help="Path to the enriched rules JSON (e.g., *_cel.json).",
    )
    parser.add_argument(
        "--num-rows", "-n",
        type=int,
        default=5,
        help="Number of mock data rows per rule (default: 5).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.rules_file):
        print(f"Error: File not found: {args.rules_file}", file=sys.stderr)
        sys.exit(1)

    # 1. Load the enriched rules JSON
    with open(args.rules_file, "r", encoding="utf-8") as f:
        rules: list[dict] = json.load(f)

    all_rule_data: Dict[str, Any] = {}

    for i, rule in enumerate(rules):
        rule_id = rule.get("rule_id", f"unknown_{i}")
        rule_name = rule.get("rule_name", "")
        calc_cel = rule.get("calculation_cel")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(rules)}] {rule_id} — {rule_name}")
        print(f"{'='*60}")

        if not calc_cel:
            print("  ⚠️  No calculation_cel found — skipping.")
            all_rule_data[rule_id] = {
                "rule_name": rule_name,
                "variable_specs": [],
                "data": [],
            }
            continue

        # 2. Extract input variables from calculation_cel (+ routing_cel)
        var_paths = extract_variables_from_cel(calc_cel)
        routing_cel = rule.get("routing_cel")
        if routing_cel:
            var_paths.update(extract_variables_from_cel(routing_cel))

        # Filter out derived.* (computed outputs, not inputs)
        input_vars = {v for v in var_paths if not v.startswith("derived.")}

        if not input_vars:
            print("  ℹ️  No input variables found in logic — skipping.")
            all_rule_data[rule_id] = {
                "rule_name": rule_name,
                "variable_specs": [],
                "data": [],
            }
            continue

        print(f"  📋 Found {len(input_vars)} input variable(s): {sorted(input_vars)}")

        # 3. Gather context from data_required
        var_contexts = gather_variable_context(rule, input_vars)

        # 4. LLM: infer Faker specs
        try:
            specs = infer_variable_specs(rule, var_contexts)
            print(f"  ✅ Inferred specs for {len(specs)} variable(s):")
            for s in specs:
                kw = s.faker_kwargs or {}
                print(f"     {s.variable_path}: {s.faker_provider}({kw}) -> {s.data_type}")
        except Exception as e:
            print(f"  ⚠️  LLM inference failed: {e}")
            specs = []

        # 5. Validate specs against CEL expression (fix type mismatches)
        if specs:
            specs = validate_specs_against_cel(
                specs, calc_cel, routing_cel=rule.get("routing_cel"),
            )

        # 6. Generate mock data
        if specs:
            flat_rows = generate_dynamic_data(specs, num_rows=args.num_rows)
            nested_rows = [nest_flat_row(row) for row in flat_rows]

            # 6b. Ensure at least one row satisfies the primary positive
            #     condition by injecting a "golden" row built from the
            #     CEL expression's literal values.
            golden_row = _build_golden_row(calc_cel, specs)
            if golden_row:
                nested_golden = nest_flat_row(golden_row)
                # Replace the first row with the golden row so there is
                # always at least one row that triggers the positive path.
                if nested_rows:
                    nested_rows[0] = nested_golden
                else:
                    nested_rows.append(nested_golden)
                print(f"  🎯 Injected golden row (row 0) to ensure positive-path coverage.")

            print(f"  📊 Generated {len(nested_rows)} data row(s).")
        else:
            nested_rows = []
            specs = []

        all_rule_data[rule_id] = {
            "rule_name": rule_name,
            "output_variable": rule.get("output_variable"),
            "variable_specs": [s.model_dump() for s in specs],
            "data": nested_rows,
        }

    # 7. Save schema to JSON
    base, ext = os.path.splitext(args.rules_file)
    schema_file = f"{base}_schema.json"

    schema_data: Dict[str, Any] = {}
    for rid, rdata in all_rule_data.items():
        schema_data[rid] = {
            "rule_name": rdata["rule_name"],
            "output_variable": rdata.get("output_variable"),
            "variable_specs": rdata["variable_specs"],
        }

    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n📐 Schema saved to: {schema_file}")

    # 8. Save mock data output
    output_file = f"{base}_mockdata{ext}"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_rule_data, f, indent=2, default=str, ensure_ascii=False)

    total_rows = sum(len(v["data"]) for v in all_rule_data.values())
    print(f"\n✅ Done! Mock data for {len(all_rule_data)} rules "
          f"({total_rows} total rows) saved to: {output_file}")
