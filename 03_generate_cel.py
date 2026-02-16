import argparse
import json
import os
import re
import sys
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from cel import evaluate as cel_evaluate, Context as CelContext
import celpy


load_dotenv()


# LLM config from environment
OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")


def _get_llm() -> ChatOpenAI:
    """Create ChatOpenAI instance from environment config."""
    kwargs = {"model": MODEL_NAME, "temperature": 0}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    return ChatOpenAI(**kwargs)


# ==========================================
# PART 0: CEL-safe identifier helpers
# ==========================================

def _sanitize_cel_identifier(name: str) -> str:
    """Convert an arbitrary string into a valid CEL identifier.

    - Replace spaces, hyphens, and other non-word characters with ``_``.
    - Prefix with ``_`` if the result starts with a digit.
    - Collapse consecutive underscores.
    """
    name = re.sub(r"[^\w]", "_", name)   # non-alphanum → _
    name = re.sub(r"_+", "_", name)      # collapse __
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "_" + name
    return name


def _extract_cel_prefix(data_source: str) -> str:
    """Derive a CEL-safe variable prefix from a data_source string.

    Strategy:
    1. If the name contains a parenthesised abbreviation like
       ``"Entity Profile System (EPS)"`` → use ``EPS``.
    2. Otherwise sanitise the full name:
       ``"SIS Enrollment"`` → ``SIS_Enrollment``.
    """
    m = re.search(r"\(([^)]+)\)", data_source)
    raw = m.group(1) if m else data_source
    return _sanitize_cel_identifier(raw)


def _preprocess_rule_for_cel(rule_object: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of *rule_object* whose ``data_required`` entries
    have been annotated with a ``cel_prefix`` field — a CEL-safe identifier
    the LLM should use as the variable namespace for that data source.
    """
    import copy
    rule = copy.deepcopy(rule_object)
    for dr in rule.get("data_required", []):
        src = dr.get("data_source", "")
        dr["cel_prefix"] = _extract_cel_prefix(src)
    return rule


# ==========================================
# PART 1: Define Pydantic Models for LLM Output
# ==========================================

class CelArtifact(BaseModel):
    rule_id: str = Field(..., description="The rule ID, same as input.")
    rule_name: str = Field(..., description="The rule name, same as input.")
    output_variable: str = Field(..., description="The variable path where the calculation result is stored (e.g., 'derived.responsible_district').")
    calculation_cel: str = Field(..., description="A valid CEL expression that calculates the business value.")
    routing_cel: Optional[str] = Field(None, description="A valid CEL expression that determines the next Rule ID, or null if this is the final step.")

# ==========================================
# PART 2: The Compute-then-Route CEL Compiler
# ==========================================
def compile_doc_to_cel(
    rule_object: Dict[str, Any],
    error_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    llm = _get_llm()

    system_prompt = """### SYSTEM ROLE
You are a Senior Rule Architect. Your task is to convert a "Canonical Rule Artifact" into a structured "Compute-then-Route" object using CEL (Common Expression Language) expressions.

### INPUT DATA SCHEMA
You will receive a JSON object with the following fields:
1. `rule_id` & `rule_name`: Identifiers.
2. `data_required`: A list of data sources. Each entry has a `cel_prefix` field — a CEL-safe variable prefix you **MUST** use for that data source's attributes (e.g., `SIS.grade_level`, `EPS.school_id`). Always prefer `cel_prefix` over inventing your own name.
3. `conditions`: A flat list of strings describing logic constraints.
   - **CRITICAL:** This list often mixes "Gatekeeper" checks (must be true to proceed) with "Mutually Exclusive" scenarios (If School A, do X; If School B, do Y). You must untangle this into a proper decision tree.
4. `action`: Describes the business calculation (e.g., "Assign District").
5. `outcome`: Describes the goal.
6. `outgoing_routes`: Possible next steps.

### OUTPUT SCHEMA
Return a JSON object with exactly these fields:

1. `rule_id`: (String) Same as input.
2. `rule_name`: (String) Same as input.
3. `output_variable`: (String) The nested path to save the result (e.g., "derived.responsible_district").
   - *Convention:* Use "derived." + snake_case description of the outcome.
4. `calculation_cel`: (String) A valid CEL expression that calculates the business value.
   - **Goal:** Calculate the BUSINESS VALUE (String, Number, Boolean).
   - **Gatekeepers:** If specific conditions (like "not null" or "not in Evaluation") are required, use a ternary that returns `null` if they fail.
   - **Branching:** If the conditions list school names, map them to their distinct results using chained ternaries.
   - **Inference:** If the input doesn't explicitly state the result value (e.g. "Paris District"), you must INFER it from the condition text (e.g. "Paris Cooperative High School" -> "Paris District").
5. `routing_cel`: (String or null) A valid CEL expression that determines the next Rule ID.
   - **Goal:** Determine the NEXT RULE ID based on the computed output.
   - **Input:** Reference the variable defined in `output_variable` using dot notation (e.g., `derived.responsible_district`).
   - **Output:** Return a String Rule ID (e.g., "R005").
   - **Default:** If `outgoing_routes` is empty or this is the final step, set to `null`.

### VARIABLE NAMING RULES  (MUST follow — violations will cause parse errors)
Every segment in a dotted variable path (e.g. `SIS.grade_level`) must be a valid CEL identifier:
- **Allowed characters:** letters, digits, and underscores only.
- **Must start with a letter or underscore** — NEVER a digit.
- **No spaces, hyphens, slashes, or special characters.**

**How to build variable names:**
1. Use the `cel_prefix` field from `data_required` as the namespace (it is already sanitised).
2. Use the `attribute_name` as the field name (it is already snake_case).
3. Combine with dot notation: `{{cel_prefix}}.{{attribute_name}}` → e.g. `SIS.grade_level`.

**Common pitfalls and their fixes:**
| Bad (parse error)                 | Good                                |
|-----------------------------------|-------------------------------------|
| `5Essentials.response_rate`       | `_5Essentials.response_rate`        |
| `SIS Enrollment.student_count`    | `SIS_Enrollment.student_count`      |
| `UC BEAR.score`                   | `UC_BEAR.score`                     |
| `Pearson AccessNext.result`       | `Pearson_AccessNext.result`         |

### SUPPORTED CEL FEATURES — COMPLETE WHITELIST
You may ONLY use the features listed below. Anything not listed here will cause a runtime error.

**Operators:**
- Comparison: `==`, `!=`, `>`, `>=`, `<`, `<=`
- Boolean logic: `&&`, `||`, `!`
- Arithmetic: `+`, `-`, `*`, `/`, `%`  (on numbers only; `+` also concatenates strings)
- Membership: `value in [list]`; negated: `!(value in [list])` (do NOT write `not in`)
- Ternary: `condition ? true_value : false_value`

**Type conversions (the ONLY functions available):**
- `double(x)` — convert int to double
- `int(x)` — convert double to int
- `string(x)` — convert to string

**String methods (called ON a string variable with dot notation):**
- `.contains('substr')`
- `.startsWith('prefix')`
- `.endsWith('suffix')`
- `size(str)` — length of a string

**Literals:**
- Strings: single quotes only → `'hello'`
- Integers: `0`, `1`, `100`
- Doubles: `0.0`, `3.14`, `100.0`
- Booleans: `true`, `false`
- Null: `null`
- Lists (for `in` checks only): `['a', 'b', 'c']` or `[1, 2, 3]`

**NOTHING ELSE EXISTS.** No other functions, methods, macros, or syntax.

### TYPE SYSTEM (CRITICAL — violations cause runtime errors)
CEL is STRICTLY typed. You CANNOT mix int and double, or number and string.

**Rule T1 — ALL arithmetic must use double():**
ANY expression involving arithmetic (`+`, `-`, `*`, `/`, `%`) with numeric values MUST wrap ALL variable operands AND integer literals in `double()`. This prevents int/double mixing.
| BAD (will crash) | GOOD |
|---|---|
| `(cond ? 1 : 0) * 100.0` | `(cond ? 1.0 : 0.0) * 100.0` |
| `VAR >= 100.0` (if VAR might be int) | `double(VAR) >= 100.0` |
| `VAR / VAR2 * 100` | `double(VAR) / double(VAR2) * 100.0` |
| `int(X) - 1` | `double(X) - 1.0` |

**Rule T2 — String concatenation requires string():**
When using `+` to build strings, EVERY non-string operand MUST be wrapped in `string()`. Variables that are numbers or booleans will crash if concatenated directly.
| BAD (will crash) | GOOD |
|---|---|
| `VAR_int + '\|' + VAR_str` | `string(VAR_int) + '\|' + VAR_str` |
| `VAR_float + '\|' + VAR_int` | `string(VAR_float) + '\|' + string(VAR_int)` |

**Rule T3 — Comparisons must match types:**
Both sides of `==`, `!=`, `>`, `>=`, `<`, `<=` MUST be the same type. If a variable might be int but the literal is double, wrap the variable: `double(VAR) >= 100.0`. If comparing with a string literal, the variable must be a string.

**Rule T4 — Ternary branches must return the same type:**
Both the true and false branch MUST produce the same type. `cond ? 'text' : null` is OK. `cond ? 1 : 'text'` will FAIL. When returning numbers, make both branches double: `cond ? 1.0 : 0.0` (not `cond ? 1 : 0` when multiplying by a double later).

**Rule T5 — Ternary conditions must be boolean expressions:**
NEVER use a bare variable, string literal, or number as a ternary condition.
| BAD (will crash) | GOOD |
|---|---|
| `SIS.has_iep ? 'Y' : 'N'` | `SIS.has_iep == true ? 'Y' : 'N'` |
| `'some text' ? X : Y` | *(remove — a string literal is always truthy)* |
| `SIS.count ? X : Y` | `SIS.count > 0 ? X : Y` |

**Rule T6 — When in doubt, use double() everywhere in numeric expressions:**
It is ALWAYS safe to wrap any numeric variable in `double()`. When an expression mixes variables, literals, and arithmetic, wrap EVERY numeric operand in `double()` and use `.0` suffixed literals:
`double(VAR) >= 100.0 ? (double(A) / double(B)) * 100.0 : 0.0`

### CEL EXPRESSION RULES
1. **Variable Access:** Dot notation only: `SIS.grade_level`, `EPS.school_id`.
2. **Every variable must be namespaced:** Use `cel_prefix.attribute_name` format. NEVER use bare names like `salary` — always `EIS.salary`.
3. **Ternary nesting:** Wrap inner ternaries in parentheses: `cond1 ? (cond2 ? valA : valB) : valC`. NEVER `cond1 ? cond2 ? valA : valB : valC`.
4. **Balanced parentheses:** Count all `(` and `)` before returning — they must match.
5. **Division zero-guard:** ALWAYS guard against division by zero: `b != 0.0 ? (double(a) / double(b)) : 0.0`.
6. **String concatenation:** Wrap non-string values in `string()`: `string(VAR_int) + '|' + string(VAR_float)`.
7. **Scalar output only:** Return a single value (string, number, boolean, or null). Encode multi-valued outputs as delimited strings.
8. **No date arithmetic:** Do NOT subtract or add dates. Compare date strings directly: `date_a >= '2024-01-01'`.
9. **Per-record only:** CEL evaluates ONE record at a time. No `sum()`, `count()`, `avg()`, `filter()`, `map()`, or ANY aggregation.
10. **Operator precedence:** `||` has lower precedence than `&&`. When mixing both, ALWAYS wrap the `||` group in parentheses. Write `A && (B || C || D)`, NEVER `A && B || C || D`.
11. **Grade-level / numeric-string ranges:** NEVER use `>=` / `<=` to compare grade levels or other numeric strings (e.g. `grade_level >= '1' && grade_level <= '12'`). String comparison is lexicographic so `'9' > '12'` is TRUE. Use an explicit `in` list instead: `grade_level in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']`.
12. **Division zero-guard structure:** Guard the ENTIRE division, not just the denominator. Write: `denominator != 0.0 ? (double(a) / double(b)) * 100.0 : 0.0`. NEVER write `a / (b != 0 ? b : 0)` — this still divides when b is zero.
13. **NEVER return null as the else-branch:** Every ternary else-branch MUST return a meaningful string value, NEVER null. Use `'Not Applicable'` when the condition doesn't apply. Write: `condition ? 'Result' : 'Not Applicable'`. NEVER write: `condition ? 'Result' : null`.
14. **String .size() returns int:** The `.size()` method on strings returns an integer. Compare it to integer literals, NOT doubles. Write: `VAR.size() > 0` — NEVER `VAR.size() > 0.0`. Also, if the CEL expression compares a variable using `.size()`, that variable must be a string; if the intent is to check it is non-empty, use `VAR != '' && VAR != null` instead.
15. **Output value consistency:** Use consistent Title Case casing for output values. Compliance statuses must always be `'Compliant'` or `'Non-Compliant'` (Title Case, with hyphen). NEVER use `'compliant'`, `'non_compliant'`, or `'non-compliant'`.
16. **No leading/trailing whitespace:** Output string values must NEVER have leading or trailing spaces. Write `'compliant'` not `' compliant'`.

### EXAMPLE INPUT
{{
  "rule_id": "R003",
  "conditions": [
    "SIS.grade_level >= 12",
    "SIS.home_school_id is 'Bismark High'"
  ],
  "action": "Assign responsible district."
}}

### EXAMPLE OUTPUT
{{
  "rule_id": "R003",
  "rule_name": "Responsible Entity Determination",
  "output_variable": "derived.responsible_district",
  "calculation_cel": "SIS.grade_level >= 12 && SIS.home_school_id == 'Bismark High' ? 'Bismark District' : 'Unassigned'",
  "routing_cel": "derived.responsible_district == 'Bismark District' ? 'R005_Bismark_Workflow' : 'R004_Standard_Workflow'"
}}

### YOUR TASK
Convert the following artifact into the Compute-then-Route CEL structure."""

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    print("🤖 COMPILER: Converting Rule Artifact to Compute-then-Route CEL...")

    # Pre-process: inject cel_prefix into each data_required entry so the
    # LLM sees clean, CEL-safe variable namespaces.
    processed_rule = _preprocess_rule_for_cel(rule_object)
    artifact_json = json.dumps(processed_rule, indent=2)

    # Build the base messages from the template
    prompt_value = final_prompt.invoke({"input": artifact_json})
    messages = prompt_value.to_messages()

    # If we have error feedback from a prior failed verification, append it
    # as a raw HumanMessage — this bypasses ChatPromptTemplate entirely so
    # any special characters in the feedback are safe.
    if error_feedback:
        messages.append(HumanMessage(content=(
            "### PREVIOUS ATTEMPT FAILED VERIFICATION\n"
            "The CEL expressions you produced in your last attempt were evaluated "
            "against test data and raised errors. Please carefully review the "
            "errors below and regenerate CORRECTED CEL expressions.\n\n"
            + error_feedback
        )))

    result = llm.with_structured_output(CelArtifact).invoke(messages)
    return {
        "rule_id": result.rule_id,
        "rule_name": result.rule_name,
        "output_variable": result.output_variable,
        "calculation_cel": result.calculation_cel,
        "routing_cel": result.routing_cel,
    }


# ==========================================
# PART 3: Verification – compile-check CEL expressions (syntax only)
# ==========================================

def _fix_unbalanced_parens(expression: str) -> str:
    """Auto-fix missing trailing parentheses — a common LLM mistake with
    deeply chained ternaries."""
    open_count = expression.count("(")
    close_count = expression.count(")")
    if open_count > close_count:
        expression += ")" * (open_count - close_count)
    return expression


# Regex: a "word" that starts with one or more digits followed by at least
# one letter/underscore (i.e. an identifier, not a bare number like 50 or 100.0).
_DIGIT_IDENT_RE = re.compile(r"(?<!\w)(\d+[a-zA-Z_]\w*)")


def _fix_digit_starting_identifiers(expression: str) -> str:
    """CEL identifiers cannot start with a digit.  Prefix them with ``_``.

    ``5Essentials.student_response_rate``  →  ``_5Essentials.student_response_rate``

    Only modifies code tokens — single-quoted string literals are left
    untouched.
    """
    # Split on single-quoted strings so we only touch code segments
    parts = re.split(r"('(?:[^'\\]|\\.)*')", expression)
    for i in range(0, len(parts), 2):          # even indices = code
        parts[i] = _DIGIT_IDENT_RE.sub(r"_\1", parts[i])
    return "".join(parts)


# ── Nested-ternary auto-fix helpers ────────────────────────────────────

def _find_toplevel(expr: str, char: str) -> int:
    """Return the index of the first *char* at the top level (outside
    parentheses and single-quoted string literals), or ``-1``."""
    depth = 0
    in_str = False
    i = 0
    while i < len(expr):
        c = expr[i]
        if in_str:
            if c == "'":
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and c == char:
            return i
        i += 1
    return -1


def _find_matching_colon(expr: str, start: int) -> int:
    """Starting from *start* (the character right after a ``?``), find the
    position of the matching ``:`` by counting nested ``?``/``:`` pairs.
    Respects parentheses and string literals."""
    ternary_depth = 1
    paren_depth = 0
    in_str = False
    i = start
    while i < len(expr):
        c = expr[i]
        if in_str:
            if c == "'":
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
            i += 1
            continue
        if c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth -= 1
        elif paren_depth == 0:
            if c == "?":
                ternary_depth += 1
            elif c == ":":
                ternary_depth -= 1
                if ternary_depth == 0:
                    return i
        i += 1
    return -1


def _is_fully_wrapped(expr: str) -> bool:
    """Return ``True`` if *expr* is entirely enclosed in one matched pair
    of parentheses (e.g. ``(A ? B : C)``)."""
    if not expr.startswith("("):
        return False
    depth = 0
    in_str = False
    for i, c in enumerate(expr):
        if in_str:
            if c == "'":
                in_str = False
            continue
        if c == "'":
            in_str = True
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i == len(expr) - 1
    return False


def _fix_nested_ternaries(expr: str) -> str:
    """Ensure nested ternary true-branches are parenthesised.

    CEL requires ``A ? (B ? C : D) : E`` but LLMs often produce
    ``A ? B ? C : D : E`` which fails to parse.  This function
    recursively wraps unparenthesised true-branches.
    """
    q_pos = _find_toplevel(expr, "?")
    if q_pos < 0:
        return expr  # no ternary at all

    true_start = q_pos + 1
    colon_pos = _find_matching_colon(expr, true_start)
    if colon_pos < 0:
        return expr  # malformed — don't touch

    true_branch = expr[true_start:colon_pos].strip()
    false_branch = expr[colon_pos + 1:]

    # ── Recursively fix the true-branch ────────────────────────────────
    if _is_fully_wrapped(true_branch):
        # Already wrapped — fix inside the parentheses
        inner = true_branch[1:-1]
        fixed_inner = _fix_nested_ternaries(inner)
        true_branch = "(" + fixed_inner + ")"
    else:
        fixed_true = _fix_nested_ternaries(true_branch)
        # If the (fixed) true-branch still contains a top-level ternary,
        # it needs to be wrapped in parens.
        if _find_toplevel(fixed_true, "?") >= 0:
            true_branch = "(" + fixed_true + ")"
        else:
            true_branch = fixed_true

    # ── Recursively fix the false-branch ───────────────────────────────
    fixed_false = _fix_nested_ternaries(false_branch)

    # ── Reconstruct ────────────────────────────────────────────────────
    condition = expr[:q_pos].rstrip()
    return condition + " ? " + true_branch + " : " + fixed_false.lstrip()


def _compile_check_cel(expression: str) -> Optional[str]:
    """Try to compile a CEL expression and return an error message if it
    has a syntax error, or ``None`` if the expression is valid.

    We call ``cel_evaluate(expr, {})`` with an empty context:
    - ``ValueError`` with "Failed to parse" → genuine syntax error → invalid.
    - ``ValueError`` with "execution error" → valid CEL that can't evaluate
      without data (e.g. ``double(SIS.score)``) → OK.
    - ``RuntimeError`` → valid CEL, just missing variables at runtime → OK.
    - No exception → expression is trivially valid.
    """
    try:
        cel_evaluate(expression, {})
        return None  # evaluated fine (e.g. purely literal expression)
    except ValueError as e:
        msg = str(e)
        # "execution error" means the CEL parsed fine but failed at runtime
        # (e.g. double() wrapping an undefined variable). That's OK.
        if "execution error" in msg.lower():
            return None
        return msg  # genuine syntax / parse error
    except RuntimeError:
        return None  # valid CEL; runtime error is expected without data


def _check_forbidden_patterns(expression: str) -> Optional[str]:
    """Check a CEL expression for forbidden patterns that are syntactically
    valid but will fail at runtime.

    Returns an error message if a forbidden pattern is found, or ``None``.
    """
    # Strip single-quoted strings to avoid false positives on string content
    cleaned = re.sub(r"'(?:[^'\\]|\\.)*'", "", expression)

    # ── 1. Aggregation / collection functions ──────────────────────────
    agg_match = re.search(
        r"\b(SUM|sum|COUNT|count|AVG|avg|MIN|min|MAX|max|AVERAGE|average"
        r"|TOTAL|total|filter|map|reduce|exists|all|flatten|sort|reverse"
        r"|distinct|unique|length|len)\s*\(",
        cleaned,
    )
    if agg_match:
        fn = agg_match.group(1)
        return (
            f"FORBIDDEN: '{fn}()' is not available in this CEL runtime. "
            f"CEL evaluates one record at a time. Express per-record logic "
            f"(e.g. a 0/1 flag or the raw value) and remove '{fn}()' calls."
        )

    # ── 2. Map / struct literals  { 'key': value }  ───────────────────
    if re.search(r"\{\s*'?\w+'?\s*:", expression):
        return (
            "FORBIDDEN: Map/struct literals {{ 'key': value }} are not supported. "
            "Return a single scalar value. "
            "Encode multi-valued outputs as a delimited string."
        )

    # ── 3. Custom / lookup functions ──────────────────────────────────
    fn_match = re.search(r"\b(lookup_\w+|get_\w+)\s*\(", cleaned)
    if fn_match:
        fn = fn_match.group(1)
        return (
            f"FORBIDDEN: Custom function '{fn}()' does not exist in CEL. "
            f"Use ternary chains or 'in' checks instead."
        )

    # ── 4. Date arithmetic (subtraction / daysBetween / datediff) ─────
    if re.search(r"_date\s*[-+]\s*\w+.*_date", cleaned):
        return (
            "FORBIDDEN: Date arithmetic on string dates is not supported. "
            "Compare date strings directly with <, >, == (ISO ordering)."
        )
    date_fn_match = re.search(
        r"\b(daysBetween|dateDiff|datediff|dateAdd|dateadd|"
        r"duration|timestamp|getDate|getMonth|getYear|"
        r"parseDate|formatDate|toDate|toTimestamp)\s*\(",
        cleaned,
    )
    if date_fn_match:
        fn = date_fn_match.group(1)
        return (
            f"FORBIDDEN: '{fn}()' is not available in this CEL runtime. "
            f"Compare date strings directly with <, >, == (ISO ordering). "
            f"Do NOT attempt date parsing, arithmetic, or timestamp functions."
        )

    # ── 5. Unsupported string methods (.split, .trim, .lower, etc.) ───
    str_method_match = re.search(
        r"\.(split|trim|strip|lower|upper|toLower|toUpper|toLowerCase"
        r"|toUpperCase|replace|replaceAll|substring|substr|slice"
        r"|matches|match|indexOf|lastIndexOf|charAt|padLeft|padRight"
        r"|format|join|concat)\s*\(",
        cleaned,
    )
    if str_method_match:
        fn = str_method_match.group(1)
        allowed = ".contains(), .startsWith(), .endsWith(), size()"
        return (
            f"FORBIDDEN: '.{fn}()' is not available in this CEL runtime. "
            f"The only supported string methods are: {allowed}. "
            f"Rewrite using those methods or ternary logic."
        )

    # ── 6. Unsupported type conversions ───────────────────────────────
    conv_match = re.search(
        r"\b(float|str|bool|list|dict|Number|String|Boolean"
        r"|parseInt|parseFloat|parseDouble|toString|toInt|toDouble"
        r"|to_int|to_double|to_string)\s*\(",
        cleaned,
    )
    if conv_match:
        fn = conv_match.group(1)
        return (
            f"FORBIDDEN: '{fn}()' does not exist in CEL. "
            f"The only type conversions available are: double(), int(), string(). "
            f"Replace '{fn}()' with the correct CEL function."
        )

    # ── 7. Math functions ─────────────────────────────────────────────
    math_match = re.search(
        r"\b(abs|ceil|floor|round|pow|sqrt|log|log10|exp|mod"
        r"|Math\.\w+|math\.\w+)\s*\(",
        cleaned,
    )
    if math_match:
        fn = math_match.group(1)
        return (
            f"FORBIDDEN: '{fn}()' is not available in CEL. "
            f"Only basic arithmetic operators (+, -, *, /, %) are supported. "
            f"Rewrite using those operators and ternary logic."
        )

    # ── 8. Bare boolean condition in ternary ──────────────────────────
    #    Catches `VAR.attr ? x : y` where the condition has no operator.
    #    Valid ternary conditions must have ==, !=, <, >, <=, >=, in, &&, ||, !
    ternary_match = re.search(r"\?", cleaned)
    if ternary_match:
        # Extract the condition part (text before the first top-level ?)
        q_pos = _find_toplevel(cleaned, "?")
        if q_pos > 0:
            condition_part = cleaned[:q_pos].strip()
            # If the condition is purely a dotted variable (or a negated one),
            # it's a bare boolean condition — likely a string masquerading as bool
            if re.fullmatch(r"!?\s*[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+", condition_part):
                return (
                    f"FORBIDDEN: Bare variable used as ternary condition: "
                    f"'{condition_part.strip()}'. CEL requires explicit boolean "
                    f"comparison. Write '{condition_part.strip()} == true' (if boolean) "
                    f"or '{condition_part.strip()} == \\'true\\'' (if string)."
                )
            # Also check within nested ternaries — look for `: VAR.attr ?`
            inner_bare = re.search(
                r":\s*(!?\s*[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*\?",
                cleaned,
            )
            if inner_bare:
                bare_cond = inner_bare.group(1).strip()
                if not re.search(r"[><=!&|]", bare_cond):
                    return (
                        f"FORBIDDEN: Bare variable used as ternary condition: "
                        f"'{bare_cond}'. CEL requires explicit boolean comparison."
                    )

    # ── 9. Map indexing on variables (some_map[key]) ──────────────────
    #    Must NOT match `value in ['a', 'b']` — that's valid CEL membership.
    bracket_matches = re.finditer(
        r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*\[", cleaned
    )
    _BRACKET_OK = {"in"}  # keywords that can precede [
    for bm in bracket_matches:
        var = bm.group(1)
        if var in _BRACKET_OK:
            continue
        # Also skip if this is a list literal context like `['a','b']`
        after_bracket = cleaned[bm.end():]
        if after_bracket.lstrip().startswith("'") or after_bracket.lstrip().startswith('"'):
            # Could be a list literal — only flag if the var is dotted (a real map access)
            if "." not in var:
                continue
        return (
            f"FORBIDDEN: Map/array indexing '{var}[...]' is not supported. "
            f"Use ternary chains instead."
        )

    # ── 10. Bare (unqualified) variable names ─────────────────────────
    _CEL_KEYWORDS = {
        "true", "false", "null", "in", "double", "int", "uint", "string",
        "bool", "bytes", "size", "has", "type",
    }
    # Strip dotted identifiers (qualified names) so only bare names remain
    no_dotted = re.sub(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b", "", cleaned)
    # Find bare identifiers adjacent to a comparison or arithmetic operator
    bare_in_op = re.findall(
        r"(?:[><=!]=?|[+\-*/%])\s*\b([a-zA-Z_]\w*)\b"
        r"|\b([a-zA-Z_]\w*)\b\s*(?:[><=!]=?|[+\-*/%])",
        no_dotted,
    )
    for groups in bare_in_op:
        name = groups[0] or groups[1]
        if not name or name in _CEL_KEYWORDS:
            continue
        if name.isdigit():
            continue
        return (
            f"FORBIDDEN: Bare variable '{name}' is not namespace-qualified. "
            f"All variables must use the 'prefix.attribute' format from "
            f"data_required (e.g., 'EIS.{name}' instead of '{name}'). "
            f"Check the cel_prefix in data_required for the correct prefix."
        )

    # ── 11. Division without double() conversion ───────────────────────
    #    Catches `VAR / VAR` where neither side is wrapped in double().
    if "/" in cleaned:
        # Find all VAR / VAR patterns
        div_match = re.search(
            r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*/\s*"
            r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b",
            cleaned,
        )
        if div_match:
            a, b = div_match.group(1), div_match.group(2)
            # Check if this division is NOT inside a double() wrapper
            # by checking the context around the match
            start = max(0, div_match.start() - 10)
            prefix = cleaned[start:div_match.start()]
            if "double(" not in prefix:
                return (
                    f"FORBIDDEN: Division '{a} / {b}' without double() conversion. "
                    f"Wrap both operands: 'double({a}) / double({b})'. "
                    f"Also guard against zero: "
                    f"'{b} != 0 ? (double({a}) / double({b})) : 0.0'."
                )

    # ── 12. Int * double mixing — (? 1 : 0) * 100.0  ─────────────────
    #    Catches integer ternary results multiplied by a double literal.
    int_times_dbl = re.search(
        r"\?\s*(\d+)\s*:\s*(\d+)\s*\)\s*\*\s*(\d+\.\d+)", cleaned
    )
    if int_times_dbl:
        i1, i2, d = int_times_dbl.group(1), int_times_dbl.group(2), int_times_dbl.group(3)
        return (
            f"TYPE ERROR: Ternary returns int ({i1}/{i2}) then multiplied by "
            f"double ({d}). CEL cannot mix int and double. "
            f"Change the ternary to return doubles: '? {i1}.0 : {i2}.0' "
            f"or wrap in double(): '? double({i1}) : double({i2})'."
        )

    # Also catch: int_literal * double_literal or vice versa outside ternary
    bare_int_dbl = re.search(
        r"(?<!\d)(?<!\.)(\d+)\s*[*/%]\s*(\d+\.\d+)", cleaned
    )
    if bare_int_dbl:
        i, d = bare_int_dbl.group(1), bare_int_dbl.group(2)
        if not i.startswith("0") or i == "0":  # skip things like 0.5 misparse
            return (
                f"TYPE ERROR: Integer literal {i} mixed with double literal {d} "
                f"in arithmetic. Use double literals everywhere: "
                f"{i}.0 instead of {i}."
            )

    # ── 13. Bare string literal as ternary condition ──────────────────
    #    Catches: `&& 'some text' ? X : Y` — a string literal used as bool.
    #    Must NOT flag: `VAR == 'val' ? X : Y` (comparison is valid).
    for str_q_match in re.finditer(r"'[^']*'\s*\?", expression):
        # Look at what's before the string literal
        before_start = str_q_match.start()
        prefix = expression[:before_start].rstrip()
        # Valid if preceded by a comparison operator: ==, !=, >=, <=, >, <
        if prefix and prefix[-1] in "=!><":
            continue
        # Valid if preceded by `in [` (membership check)
        if prefix.endswith("in [") or prefix.endswith("in["):
            continue
        # Valid if inside a list literal like ['a', 'b']
        if prefix and prefix[-1] == ",":
            continue
        if prefix and prefix[-1] == "[":
            continue
        # Otherwise it's a bare string as condition — forbidden
        return (
            "TYPE ERROR: A string literal is used as a ternary condition. "
            "String literals are not booleans. Remove the ternary or "
            "replace with a proper boolean condition like "
            "'VAR == value ? ... : ...'."
        )

    # ── 14. Variable compared to double literal without double() ──────
    #    Catches: VAR >= 100.0 or VAR <= 0.5 where VAR might be int
    var_cmp_dbl = re.search(
        r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*([><=!]=?)\s*(\d+\.\d+)",
        cleaned,
    )
    if var_cmp_dbl:
        var = var_cmp_dbl.group(1)
        op = var_cmp_dbl.group(2)
        lit = var_cmp_dbl.group(3)
        # Check if the variable is NOT already wrapped in double()
        start = max(0, var_cmp_dbl.start() - 10)
        prefix = cleaned[start:var_cmp_dbl.start()]
        if "double(" not in prefix:
            return (
                f"TYPE ERROR: '{var} {op} {lit}' — variable may be int but "
                f"compared to double {lit}. Wrap in double(): "
                f"'double({var}) {op} {lit}'."
            )

    # ── 15. String range comparison on numeric values ──────────────────
    #    Catches: VAR >= '1' && VAR <= '12' — lexicographic comparison
    #    fails for multi-digit string numbers (e.g. '9' > '12').
    str_range = re.search(
        r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*>=\s*'(\d+)'\s*&&\s*"
        r"\1\s*<=\s*'(\d+)'",
        expression,  # use original (not cleaned) to see quotes
    )
    if str_range:
        var = str_range.group(1)
        low = str_range.group(2)
        high = str_range.group(3)
        return (
            f"LOGIC ERROR: String range comparison "
            f"'{var} >= \\'{low}\\' && {var} <= \\'{high}\\'' "
            f"uses lexicographic ordering, which fails for multi-digit "
            f"numbers (e.g., '9' > '12'). "
            f"Use 'in' with an explicit list instead: "
            f"'{var} in [\\'{low}\\', ..., \\'{high}\\']'."
        )

    # ── 16. Unguarded division that can produce infinity ───────────────
    #    Catches: EXPR / (cond ? N : 0.0) — denominator can be 0.0
    #    These produce +inf or -inf which are meaningless results.
    div_zero_pat = re.search(
        r"/\s*\([^)]*\?\s*[\d.]+\s*:\s*0(?:\.0)?\s*\)",
        cleaned,
    )
    if div_zero_pat:
        return (
            "LOGIC ERROR: Division by an expression that can return 0.0. "
            "The denominator ternary can evaluate to 0.0, producing infinity. "
            "Wrap the ENTIRE division in a zero guard: "
            "'(denominator != 0.0) ? (numerator / denominator) * 100.0 : 0.0'. "
            "Do NOT put the guard inside the denominator."
        )

    # ── 17. Mixed && and || without grouping parentheses ───────────────
    #    Catches: A && B.method() || C.method() — the || breaks out of &&
    #    This is almost always a precedence bug.
    #    Look for: && EXPR.method() || EXPR.method()
    if "&&" in cleaned and "||" in cleaned:
        # Find cases where || appears after && at the same paren depth
        # without the || group being wrapped in parens
        mixed_match = re.search(
            r"&&\s*[^()]*?\.\w+\([^)]*\)\s*\|\|",
            cleaned,
        )
        if mixed_match:
            return (
                "LOGIC ERROR: Mixed '&&' and '||' without grouping parentheses. "
                "'||' has lower precedence than '&&', so "
                "'A && B || C' means '(A && B) || C', NOT 'A && (B || C)'. "
                "Wrap the '||' alternatives in parentheses: "
                "'A && (B || C || D)'."
            )

    return None


# ── Post-generation test-execution validation ──────────────────────────

def _extract_variables_from_cel(expression: str) -> List[str]:
    """Extract all dotted variable references from a CEL expression.

    Returns a list of qualified variable names like ``['SIS.grade_level',
    'EPS.school_id']``.
    """
    # Find all word.word patterns (at least two segments)
    candidates = re.findall(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b", expression)
    # Strip known CEL method suffixes
    _CEL_METHODS = {
        "contains", "startsWith", "endsWith", "size", "matches",
    }
    cleaned = []
    for var in candidates:
        parts = var.split(".")
        # Remove trailing CEL method names
        while len(parts) > 1 and parts[-1] in _CEL_METHODS:
            parts = parts[:-1]
        if len(parts) >= 2:
            cleaned.append(".".join(parts))
    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order


def _generate_sample_data(
    variables: List[str],
    expression: str,
) -> Dict[str, Any]:
    """Generate deterministic sample data for a list of dotted variable paths.

    Inspects the CEL *expression* to infer the expected type for each variable
    (string, number, or boolean) and produces appropriate sample values.
    """
    # Build a lookup of string-compared variables
    str_compared = set()
    num_compared = set()
    bool_compared = set()
    # Variables compared against string literals: VAR == 'xxx'
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*[!=]=\s*'", expression
    ):
        str_compared.add(m.group(1))
    for m in re.finditer(
        r"'\s*[!=]=\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)", expression
    ):
        str_compared.add(m.group(1))
    # Variables compared against numbers: VAR >= 5 or VAR == 100
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*[><=!]=?\s*\d", expression
    ):
        num_compared.add(m.group(1))
    for m in re.finditer(
        r"\d\s*[><=!]=?\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)", expression
    ):
        num_compared.add(m.group(1))
    # Variables inside double(): double(VAR) — numeric
    for m in re.finditer(
        r"\bdouble\s*\(\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*\)", expression
    ):
        num_compared.add(m.group(1))
    # Variables compared against true/false
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*==\s*(true|false)\b", expression
    ):
        bool_compared.add(m.group(1))
    # Variables using .contains/.startsWith/.endsWith — must be strings
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*\.\s*(?:contains|startsWith|endsWith)\s*\(",
        expression,
    ):
        str_compared.add(m.group(1))
    # Variables used with `in ['list']` — infer type from list elements
    for m in re.finditer(
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s+in\s+\[", expression
    ):
        str_compared.add(m.group(1))  # most `in` checks are string lists

    data: Dict[str, Any] = {}
    for var in variables:
        parts = var.split(".")
        # Choose sample value based on inferred type
        if var in str_compared:
            value = "sample_value"
        elif var in bool_compared:
            value = True
        elif var in num_compared:
            value = 10  # non-zero to avoid division-by-zero
        else:
            # Default: try string first (most common in business rules)
            value = "sample_value"

        # Build nested dict
        current = data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    return data


def _normalize_keys_for_celpy(obj: Any) -> Any:
    """Recursively sanitise dictionary keys for celpy evaluation."""
    if isinstance(obj, dict):
        return {
            _sanitize_cel_identifier(k): _normalize_keys_for_celpy(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize_keys_for_celpy(item) for item in obj]
    return obj


def _test_execute_cel(expression: str) -> Optional[str]:
    """Test-execute a CEL expression against auto-generated sample data.

    Uses the **celpy** library (the same one ``05_execute_rules.py`` uses)
    for strict validation.  Returns an error message if execution fails,
    or ``None`` on success.
    """
    if not expression or expression.strip() == "null":
        return None

    # 1. Extract variables and generate sample data
    variables = _extract_variables_from_cel(expression)
    sample_data = _generate_sample_data(variables, expression)

    # 2. Also add derived path values for routing expressions
    if "derived" in expression:
        derived_vars = re.findall(r"derived\.(\w+)", expression)
        for dv in derived_vars:
            sample_data.setdefault("derived", {})[dv] = "sample_value"

    # 3. Execute using celpy (strict library used by execution engine)
    try:
        env = celpy.Environment()
        ast = env.compile(expression)
        prg = env.program(ast)
        activation = celpy.json_to_cel(_normalize_keys_for_celpy(sample_data))
        result = prg.evaluate(activation)
        return None  # execution succeeded
    except celpy.CELEvalError as e:
        err_str = str(e)
        lower = err_str.lower()
        # Ignore benign runtime errors caused by sample data limitations:
        # - division by zero (data issue)
        # - missing keys (minimal sample data)
        # - type mismatches from _?_:_ overload (sample type != real type)
        # - no matching overload for comparisons (sample type wrong)
        if "divide by zero" in lower or "zerodivisionerror" in lower:
            return None
        if "no such key" in lower:
            return None
        if "no matching overload" in lower:
            return None
        if "unexpected" in lower and "?" in lower and ":" in lower:
            return None
        # Real runtime error — the CEL expression itself is broken
        return f"RUNTIME ERROR (test-execute with sample data): {err_str}"
    except celpy.CELParseError as e:
        return f"SYNTAX ERROR: {e}"
    except Exception as e:
        err_str = str(e)
        lower = err_str.lower()
        if "divide by zero" in lower or "zerodivisionerror" in lower:
            return None
        if "no such key" in lower:
            return None
        # "Failed to parse" from the other cel library
        if "failed to parse" in lower:
            return f"SYNTAX ERROR: {err_str}"
        return f"RUNTIME ERROR (test-execute with sample data): {err_str}"


def _fix_null_else_branches(expression: str) -> str:
    """Replace ``null`` in ternary else-branches with ``'Not Applicable'``.

    Catches patterns like ``condition ? 'Result' : null`` and replaces
    the ``null`` with ``'Not Applicable'`` so every execution produces a
    meaningful derived output value.
    """
    if not expression:
        return expression
    # Pattern: `: null` at end of expression or before `)` — ternary else branch
    fixed = re.sub(r":\s*null\s*$", ": 'Not Applicable'", expression)
    fixed = re.sub(r":\s*null\s*\)", ": 'Not Applicable')", fixed)
    return fixed


def _fix_whitespace_in_strings(expression: str) -> str:
    """Strip leading/trailing whitespace from string literals in CEL.

    Catches patterns like ``' compliant'`` and fixes to ``'compliant'``.
    """
    if not expression:
        return expression

    def _strip_str_literal(m: re.Match) -> str:
        content = m.group(1)
        stripped = content.strip()
        return f"'{stripped}'"

    return re.sub(r"'([^']*)'", _strip_str_literal, expression)


def _fix_size_double_comparison(expression: str) -> str:
    """Fix ``.size() > 0.0`` to ``.size() > 0``.

    The CEL ``.size()`` method returns an integer, so comparing against
    a double literal causes a type mismatch.
    """
    if not expression:
        return expression
    # Fix .size() compared to double literals
    fixed = re.sub(
        r"\.size\(\)\s*([><=!]=?)\s*(\d+)\.0\b",
        lambda m: f".size() {m.group(1)} {m.group(2)}",
        expression,
    )
    return fixed


def _fix_type_mixing(expression: str) -> str:
    """Auto-fix common type-mixing issues in CEL expressions.

    This function applies safe, deterministic fixes:
    - Convert integer ternary results to doubles when multiplied by a double
    - Wrap variables in double() when compared to double literals
    - Wrap numeric variables in string() when concatenated with strings
    """
    if not expression:
        return expression

    fixed = expression

    # ── Fix A: ? int : int) * double  →  ? double : double) * double ──
    # Pattern: `? 1 : 0) * 100.0` or `? 1 : 0) / 100.0`
    def _int_ternary_to_double(m):
        i1, i2, op, d = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"? {i1}.0 : {i2}.0) {op} {d}"

    fixed = re.sub(
        r"\?\s*(\d+)\s*:\s*(\d+)\s*\)\s*([*/%])\s*(\d+\.\d+)",
        _int_ternary_to_double,
        fixed,
    )

    # ── Fix B: VAR >= 100.0  →  double(VAR) >= 100.0 ─────────────────
    # Only when VAR is a dotted identifier not already in double()
    def _wrap_var_cmp_double(m):
        pre = m.group(1) or ""
        var = m.group(2)
        op = m.group(3)
        lit = m.group(4)
        # Don't wrap if already inside double()
        if pre.rstrip().endswith("double("):
            return m.group(0)
        return f"{pre}double({var}) {op} {lit}"

    fixed = re.sub(
        r"((?:^|[^a-zA-Z_\w]))"                          # non-identifier prefix
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)"             # dotted var
        r"\s*([><=!]=?)\s*"                               # comparison operator
        r"(\d+\.\d+)",                                    # double literal
        _wrap_var_cmp_double,
        fixed,
    )

    # ── Fix C: Also fix double_literal OP VAR  →  double_literal OP double(VAR)
    def _wrap_var_cmp_double_rev(m):
        lit = m.group(1)
        op = m.group(2)
        var = m.group(3)
        post = m.group(4) or ""
        if post.lstrip().startswith(")"):
            return m.group(0)  # already inside something
        return f"{lit} {op} double({var}){post}"

    fixed = re.sub(
        r"(\d+\.\d+)\s*([><=!]=?)\s*"
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)"
        r"(\s*)",
        _wrap_var_cmp_double_rev,
        fixed,
    )

    # ── Fix D: int(VAR) in arithmetic  →  double(VAR) ────────────────
    # Pattern: int(VAR) - 1  or  int(VAR) + 1
    # int() returns IntType which can't mix with other types
    fixed = re.sub(
        r"\bint\(\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\s*\)\s*([+\-*/])\s*(\d+)(?!\.)",
        lambda m: f"double({m.group(1)}) {m.group(2)} {m.group(3)}.0",
        fixed,
    )

    # ── Fix F: Bare integer literals in comparisons → double literals ──
    # After numeric coercion, all data values are DoubleType.
    # CEL integer literals are IntType. celpy can't compare Double==Int.
    # Convert: `VAR == 2019` → `VAR == 2019.0`, `VAR != 0` → `VAR != 0.0`
    # Must not touch integers inside string quotes or list brackets with strings.
    def _int_literal_to_double(m):
        op = m.group(1)
        num = m.group(2)
        after = m.group(3) or ""
        # Don't convert if this is already a double (has decimal point after)
        if after.startswith("."):
            return m.group(0)
        return f"{op} {num}.0{after}"

    # Pattern: comparison_op SPACE integer (not followed by . which would make it a double)
    fixed = re.sub(
        r"([><=!]=?)\s*(\d+)(\s|[)\]&|?:,])",
        _int_literal_to_double,
        fixed,
    )
    # Also handle end-of-string case
    fixed = re.sub(
        r"([><=!]=?)\s*(\d+)$",
        lambda m: f"{m.group(1)} {m.group(2)}.0",
        fixed,
    )

    # ── Fix G: Integers in ternary return positions → double ──────────
    # `? 2017 :` → `? 2017.0 :` and `: 2019 )` → `: 2019.0 )`
    # Converts integer results to doubles for type consistency.
    def _ternary_int_to_double(m):
        prefix = m.group(1)  # ? or :
        num = m.group(2)
        suffix = m.group(3)  # : or ) or end
        return f"{prefix} {num}.0 {suffix}"

    fixed = re.sub(
        r"(\?)\s*(\d+)(?!\.\d)\s*(:)",
        _ternary_int_to_double,
        fixed,
    )
    fixed = re.sub(
        r"(:)\s*(\d+)(?!\.\d)\s*([)\s]|$)",
        _ternary_int_to_double,
        fixed,
    )

    # ── Fix H: Integers in list literals → doubles ────────────────────
    # `in [2020, 2022, 2024]` → `in [2020.0, 2022.0, 2024.0]`
    # Only for numeric lists (not string lists like ['a', 'b'])
    def _fix_int_list(m):
        bracket_content = m.group(1)
        # Check if it's a numeric list (no quotes)
        if "'" in bracket_content or '"' in bracket_content:
            return m.group(0)  # string list — leave alone
        # Convert each integer to double
        fixed_content = re.sub(
            r"\b(\d+)\b(?!\.)",
            lambda n: f"{n.group(1)}.0",
            bracket_content,
        )
        return f"[{fixed_content}]"

    fixed = re.sub(
        r"\[([^\]]+)\]",
        _fix_int_list,
        fixed,
    )

    # ── Fix E: Wrap non-string vars in string() for concatenation ─────
    # Pattern: VAR + '|'  or  '|' + VAR  where VAR isn't already in string()
    # This is trickier — we need to detect concatenation context.
    # Look for dotted vars adjacent to string concat patterns.
    # Only fix if there's a string literal nearby (indicating concat context)
    if "+" in fixed and "'" in fixed:
        # Find all: VAR + '  (dotted var followed by + string literal)
        for m in re.finditer(
            r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b"
            r"(\s*\+\s*')",
            fixed,
        ):
            var = m.group(1)
            # Check this isn't already wrapped in string()
            start = max(0, m.start() - 10)
            pre = fixed[start:m.start()]
            if "string(" not in pre:
                old = m.group(0)
                new = f"string({var}){m.group(2)}"
                fixed = fixed.replace(old, new, 1)

        # Also: ' + VAR  →  ' + string(VAR)
        for m in re.finditer(
            r"('\s*\+\s*)"
            r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b",
            fixed,
        ):
            var = m.group(2)
            # Skip if VAR is followed by a string method call (it's already a string)
            post_start = m.end()
            post = fixed[post_start:post_start + 15]
            if re.match(r"\s*\.\s*(?:contains|startsWith|endsWith)", post):
                continue
            # Check it's not already inside string()
            full_prefix = m.group(1)
            if "string(" in full_prefix:
                continue
            old = m.group(0)
            new = f"{m.group(1)}string({var})"
            fixed = fixed.replace(old, new, 1)

    return fixed


def verify_cel(compiled_rule: Dict[str, Any]) -> Dict[str, Any]:
    """Verify ``calculation_cel`` and ``routing_cel`` by compile-checking
    them for valid CEL syntax and rejecting forbidden patterns.

    Returns a dict with ``calculation_error`` and ``routing_error`` keys
    (each ``None`` if valid, or an error string if the syntax is bad).
    """
    calc_cel = compiled_rule.get("calculation_cel")
    route_cel = compiled_rule.get("routing_cel")
    rule_id = compiled_rule.get("rule_id", "?")
    rule_name = compiled_rule.get("rule_name", "?")

    # Auto-fix common LLM mistakes
    for field in ("calculation_cel", "routing_cel"):
        original = compiled_rule.get(field)
        if not original:
            continue
        fixed = original

        # Fix 1: identifiers starting with a digit  (e.g. 5Essentials → _5Essentials)
        fixed = _fix_digit_starting_identifiers(fixed)
        # Fix 2: nested ternaries in the true-branch need parentheses
        fixed = _fix_nested_ternaries(fixed)
        # Fix 3: missing trailing parentheses
        fixed = _fix_unbalanced_parens(fixed)
        # Fix 4: type-mixing issues (int*double, var cmp double, string concat)
        fixed = _fix_type_mixing(fixed)
        # Fix 5: null else-branches → 'Not Applicable'
        if field == "calculation_cel":
            fixed = _fix_null_else_branches(fixed)
        # Fix 6: leading/trailing whitespace in string literals
        fixed = _fix_whitespace_in_strings(fixed)
        # Fix 7: .size() > 0.0 → .size() > 0
        fixed = _fix_size_double_comparison(fixed)

        if fixed != original:
            label = field.replace("_cel", "")
            print(f"   🔧 Auto-fixed {label} CEL expression")
            compiled_rule[field] = fixed

    calc_cel = compiled_rule.get("calculation_cel")
    route_cel = compiled_rule.get("routing_cel")

    print(f"\n🔍 VERIFY: {rule_id} – {rule_name}")
    print(f"   calculation_cel: {calc_cel}")
    if route_cel:
        print(f"   routing_cel:     {route_cel}")

    # ── Layer 1: Forbidden pattern check (fast, regex-based) ──────────
    calc_error = _check_forbidden_patterns(calc_cel) if calc_cel else None
    route_error = _check_forbidden_patterns(route_cel) if route_cel else None

    # ── Layer 2: Syntax / compile check ────────────────────────────────
    if not calc_error:
        calc_error = _compile_check_cel(calc_cel) if calc_cel else "Empty expression"
    if not route_error:
        route_error = _compile_check_cel(route_cel) if route_cel else None

    # ── Layer 3: Test-execute against sample data ──────────────────────
    if not calc_error:
        calc_error = _test_execute_cel(calc_cel) if calc_cel else None
        if calc_error:
            print(f"   🧪 calculation_cel test-execution failed")
    if not route_error:
        route_error = _test_execute_cel(route_cel) if route_cel else None
        if route_error:
            print(f"   🧪 routing_cel test-execution failed")

    # ── Print results ──────────────────────────────────────────────────
    if calc_error:
        print(f"   ❌ calculation_cel ERROR: {calc_error}")
    else:
        print(f"   ✅ calculation_cel: valid + test-executed OK")

    if route_cel:
        if route_error:
            print(f"   ❌ routing_cel ERROR: {route_error}")
        else:
            print(f"   ✅ routing_cel: valid + test-executed OK")
    print()

    return {
        "calculation_error": calc_error,
        "routing_error": route_error,
    }


def _build_error_feedback(
    compiled_rule: Dict[str, Any],
    verification_result: Dict[str, Any],
) -> str:
    """Build a human-readable error report from a failed compile check."""
    lines: List[str] = []

    lines.append("#### Previously Generated CEL Expressions")
    lines.append(f"calculation_cel: {compiled_rule.get('calculation_cel')}")
    if compiled_rule.get("routing_cel"):
        lines.append(f"routing_cel: {compiled_rule.get('routing_cel')}")
    lines.append("")

    lines.append("#### Errors")
    calc_err = verification_result.get("calculation_error")
    route_err = verification_result.get("routing_error")
    if calc_err:
        lines.append(f"- calculation_cel ERROR: {calc_err}")
    if route_err:
        lines.append(f"- routing_cel ERROR: {route_err}")
    lines.append("")
    lines.append(
        "Please fix the CEL expressions so they compile AND execute without errors.\n"
        "CRITICAL RULES:\n"
        "1. ONLY use supported operators: ==, !=, >, >=, <, <=, &&, ||, !, +, -, *, /, %, in, ?:\n"
        "2. ONLY use supported functions: double(), int(), string(), size()\n"
        "3. ONLY use supported string methods: .contains(), .startsWith(), .endsWith()\n"
        "4. NO OTHER functions, methods, or macros exist — no sum(), count(), avg(), min(), max(), "
        "split(), trim(), lower(), upper(), daysBetween(), abs(), round(), filter(), map().\n"
        "5. All variables MUST be namespace-qualified: prefix.attribute (e.g. SIS.grade_level).\n"
        "6. Wrap division in double(): b != 0 ? (double(a) / double(b)) : 0.0\n"
        "7. Ternary true-branch ternaries MUST be in parentheses: A ? (B ? C : D) : E\n"
        "8. Boolean conditions must be explicit: X == true or X == 'true', not bare X.\n"
        "9. Both branches of a ternary must return the SAME type."
    )
    return "\n".join(lines)


def _has_verification_errors(result: Dict[str, Any]) -> bool:
    """Return True if the compile check found any errors."""
    return bool(result.get("calculation_error") or result.get("routing_error"))


# ==========================================
# PART 4: Compile-and-Verify with retry loop
# ==========================================

MAX_RETRIES = 5


def compile_and_verify(
    rule_object: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Compile a rule to CEL, verify syntax, and retry on failure.

    Calls ``compile_doc_to_cel``, then ``verify_cel`` (compile-check only).
    If the CEL has syntax errors the compilation is retried with the error
    details fed back into the LLM prompt, up to ``MAX_RETRIES`` attempts.

    Returns the compiled result dict on success.
    Raises RuntimeError if all attempts fail.
    """
    rule_id = rule_object.get("rule_id", "?")
    error_feedback: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        # --- compile ---
        print(f"\n{'—'*40}")
        print(f"   Attempt {attempt}/{MAX_RETRIES} for {rule_id}")
        print(f"{'—'*40}")

        try:
            result = compile_doc_to_cel(rule_object, error_feedback=error_feedback)
        except Exception as e:
            print(f"⚠️  Compilation exception on attempt {attempt}: {e}")
            error_feedback = f"Compilation raised an exception: {e}"
            continue

        # --- verify (syntax check only) ---
        verification = verify_cel(result)

        if not _has_verification_errors(verification):
            print(f"✅ Rule {rule_id} passed verification on attempt {attempt}.")
            return result

        # --- build feedback for next attempt ---
        print(f"❌ Rule {rule_id} FAILED verification on attempt {attempt}. Retrying...")
        error_feedback = _build_error_feedback(result, verification)

    print(f"🚫 Rule {rule_id} failed verification after {MAX_RETRIES} attempts. Skipping.")
    return None


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read a rules JSON file, compile each rule to CEL expressions, and save the enriched output."
    )
    parser.add_argument("rules_file", help="Path to the rules JSON file (e.g. rules.json).")
    args = parser.parse_args()

    if not os.path.isfile(args.rules_file):
        print(f"Error: File not found: {args.rules_file}", file=sys.stderr)
        sys.exit(1)

    # 1. Load the rules JSON
    with open(args.rules_file, "r", encoding="utf-8") as f:
        rules = json.load(f)

    # 2. Process each rule item
    enriched_rules = []
    failed_rules = []
    for i, rule in enumerate(rules):
        rule_id = rule.get("rule_id", f"unknown_{i}")
        rule_name = rule.get("rule_name", "")
        print(f"\n{'='*60}")
        print(f"Processing rule {i+1}/{len(rules)}: {rule_id} - {rule_name}")
        print(f"{'='*60}")

        # Compile to CEL → syntax check → test-execute → retry on failure
        result = compile_and_verify(rule)

        # Insert the compiled output into a copy of the rule item
        enriched_rule = dict(rule)
        if result:
            enriched_rule["output_variable"] = result.get("output_variable")
            enriched_rule["calculation_cel"] = result.get("calculation_cel")
            enriched_rule["routing_cel"] = result.get("routing_cel")
            enriched_rule["manual_review"] = False
        else:
            failed_rules.append(rule_id)
            enriched_rule["manual_review"] = True
            # Check if this is a non-computational rule (no data_required)
            if not rule.get("data_required"):
                enriched_rule["skip_reason"] = "non_computational_formatting_rule"
                print(f"  ℹ️  Rule {rule_id} has no data_required — flagged as non-computational formatting rule")
            else:
                enriched_rule["skip_reason"] = "cel_generation_failed"
                print(f"  ℹ️  Rule {rule_id} flagged for manual review — qualitative rule that could not be automated")
        enriched_rules.append(enriched_rule)

    # 3. Build output filename: <original_name>_cel.json
    base, ext = os.path.splitext(args.rules_file)
    output_file = f"{base}_cel{ext}"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_rules, f, indent=2, ensure_ascii=False)

    passed = len(enriched_rules) - len(failed_rules)
    print(f"\n✅ Done! {passed}/{len(enriched_rules)} rules compiled successfully → {output_file}")
    if failed_rules:
        print(f"⚠️  {len(failed_rules)} rules failed after {MAX_RETRIES} attempts: {', '.join(failed_rules)}")
        print(f"   These rules are included in the output but without CEL expressions.")
