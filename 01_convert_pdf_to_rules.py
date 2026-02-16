"""
Convert a PDF to Markdown, extract structured business rules, and discover routes.

Usage:
    python 01_convert_pdf_to_md.py <input_pdf>

The script will:
  1. Convert <input_pdf> to Markdown        (same stem, .md extension)
  2. Extract business rules from the Markdown via an LLM
     and save them as JSON                  (same stem, .json extension)
  3. Discover outgoing routes between rules
     and save the routed rules as JSON      (same stem, _routed.json extension)

Example:
    python 01_convert_pdf_to_md.py Public-Business-Rules-2024-Report-Card-Metrics.pdf
    # produces:
    #   Public-Business-Rules-2024-Report-Card-Metrics.md
    #   Public-Business-Rules-2024-Report-Card-Metrics.json
    #   Public-Business-Rules-2024-Report-Card-Metrics_routed.json

Requirements:
    pip install 'markitdown[all]' langchain-openai python-dotenv pydantic
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from markitdown import MarkItDown
from pydantic import BaseModel, Field

load_dotenv()

# ── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Pydantic models matching the rules.json schema ─────────────────────────

class OutgoingRoute(BaseModel):
    """A conditional edge leading to another rule."""
    condition: str = Field(description="The condition under which this route is taken.")
    next_rule: Optional[str] = Field(
        default=None,
        description="The rule_id of the next rule, or null if this is a terminal route.",
    )


DataAttributeType = Literal["string", "integer", "float", "boolean", "date"]


class DataAttribute(BaseModel):
    """A single data attribute within a data source."""
    attribute_name: str = Field(
        description="Snake_case field name (e.g. 'is_public_school', 'home_enrollment_count')."
    )
    description: str = Field(
        description="Brief description of what this attribute represents."
    )
    example_values: list[str] = Field(
        description="Two or more realistic example values for this attribute."
    )
    data_type: DataAttributeType = Field(
        default="string",
        description="Primitive type of the attribute: string, integer, float, boolean, or date.",
    )


class DataRequirement(BaseModel):
    """A data source and the specific attributes needed from it."""
    data_source: str = Field(
        description="Name of the data source system (e.g. 'Entity Profile System (EPS)')."
    )
    data_attributes: list[DataAttribute] = Field(
        description="The specific attributes required from this data source for the rule."
    )


class Rule(BaseModel):
    """A single business rule extracted from the source document."""
    rule_id: str = Field(description="Unique identifier, e.g. R001, R002, …")
    chunk_id: Optional[int] = Field(
        default=None,
        description="The 1-based chunk index this rule was extracted from.",
    )
    rule_name: str = Field(description="Short descriptive name of the rule.")
    entity_applied: str = Field(
        description="The entity the rule applies to, e.g. 'school', 'student', 'district'."
    )
    data_required: list[DataRequirement] = Field(
        description=(
            "Data sources and their specific attributes needed to evaluate this rule. "
            "Each entry names a data source and lists the exact attributes "
            "(with example values) required from it."
        )
    )
    conditions: list[str] = Field(
        description=(
            "List of conditions that must be checked. Each condition MUST "
            "reference the exact data source and attribute name, e.g. "
            "'EPS.is_public_school == true', "
            "'SIS.home_enrollment_count >= 10'."
        )
    )
    action: str = Field(
        description=(
            "What action is performed when the rule is evaluated. "
            "MUST reference exact data source and attribute names."
        )
    )
    outcome: str = Field(
        description="The result produced by applying the rule."
    )
    outgoing_routes: list[OutgoingRoute] = Field(
        default_factory=list,
        description="Conditional edges to subsequent rules.",
    )
    is_final: bool = Field(
        description="True if this rule has no outgoing routes and is a terminal node.",
    )


class RuleSet(BaseModel):
    """A collection of extracted business rules."""
    rules: list[Rule] = Field(description="All business rules extracted from the document.")


class RoutedRule(BaseModel):
    """A rule enriched with discovered outgoing routes."""
    rule_id: str = Field(description="The rule_id of the source rule.")
    rule_name: str = Field(description="Short descriptive name of the rule.")
    entity_applied: str = Field(description="The entity the rule applies to.")
    data_required: list[DataRequirement] = Field(
        description="Data sources and their specific attributes needed."
    )
    conditions: list[str] = Field(
        description="Conditions referencing exact data_source.attribute names."
    )
    action: str = Field(description="Action performed, referencing exact attributes.")
    outcome: str = Field(description="Result produced.")
    outgoing_routes: list[OutgoingRoute] = Field(
        default_factory=list,
        description=(
            "Discovered conditional edges to subsequent rules.  "
            "next_rule is the rule_id of the target, or 'EXTERNAL_MISSING' "
            "if the implied target is not present in the rule set."
        ),
    )
    is_final: bool = Field(
        description="True if this rule has no outgoing routes (terminal node).",
    )


class RoutedRuleSet(BaseModel):
    """The complete rule set with outgoing routes populated."""
    rules: list[RoutedRule] = Field(
        description="All business rules with their outgoing_routes and is_final fields populated.",
    )


def convert_pdf_to_markdown(input_pdf: str, output_md: str | None = None) -> str:
    """Convert a PDF file to Markdown and save the result.

    Args:
        input_pdf: Path to the input PDF file.
        output_md: Path to the output Markdown file. If None, uses the
                   input filename with a .md extension.

    Returns:
        The path to the generated Markdown file.

    Raises:
        FileNotFoundError: If the input PDF does not exist.
        markitdown.UnsupportedFormatException: If the file format is not supported.
        markitdown.FileConversionException: If conversion fails.
    """
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_pdf}")

    if output_md is None:
        output_md = str(input_path.with_suffix(".md"))

    md = MarkItDown()
    result = md.convert(str(input_path))

    Path(output_md).write_text(result.text_content, encoding="utf-8")
    return output_md


# ── Few-shot example for the extraction prompt ─────────────────────────────

_EXAMPLE_SOURCE = """\
Grades Served
SY 2024

Page 6

Grades Served
Definition
The grade level of education that a school and/or district provide for general education.

Guidance and Citation
State Statute/Guidance: N/A
Federal Statute/Guidance: EdFacts FS039 — Grades Offered

Business Rules
•  Grades Served

o  Use grades from EPS to determine Grades Served for schools.

Formula (calculations)
N/A

Sources of Data
Student Information System (SIS)
Entity Profile System (EPS)
"""

_EXAMPLE_OUTPUT = RuleSet(rules=[
    Rule(
        rule_id="R001",
        rule_name="Grades Served",
        entity_applied="school",
        data_required=[
            DataRequirement(
                data_source="Entity Profile System (EPS)",
                data_attributes=[
                    DataAttribute(
                        attribute_name="school_id",
                        description="Unique identifier for the school",
                        example_values=["SCH00012345", "SCH00067890"],
                        data_type="string",
                    ),
                    DataAttribute(
                        attribute_name="is_public_school",
                        description="Whether the school is a public school",
                        example_values=["true", "false"],
                        data_type="boolean",
                    ),
                    DataAttribute(
                        attribute_name="grades_offered",
                        description="Grade levels offered by the school",
                        example_values=["PK-12", "K-8", "9-12"],
                        data_type="string",
                    ),
                ],
            ),
            DataRequirement(
                data_source="Student Information System (SIS)",
                data_attributes=[
                    DataAttribute(
                        attribute_name="student_id",
                        description="Unique student identifier",
                        example_values=["S0000012345", "S0000067890"],
                        data_type="string",
                    ),
                    DataAttribute(
                        attribute_name="school_id",
                        description="School the student is enrolled in",
                        example_values=["SCH00012345", "SCH00067890"],
                        data_type="string",
                    ),
                    DataAttribute(
                        attribute_name="grade_level",
                        description="Grade level of the student",
                        example_values=["PK", "K", "1", "5", "12"],
                        data_type="string",
                    ),
                ],
            ),
        ],
        conditions=[
            "EPS.is_public_school == 'true' (school must be a public school)",
            "Use EPS.grades_offered to determine grades served for schools",
        ],
        action="Retrieve EPS.grades_offered for the school identified by EPS.school_id.",
        outcome="Grade levels served by the school (e.g., PK–12).",
        outgoing_routes=[
            OutgoingRoute(condition="Grades determined", next_rule="R002"),
        ],
        is_final=False,
    ),
])


# ── Rule extraction ─────────────────────────────────────────────────────────

def _build_extraction_llm(max_tokens: int | None = None):
    """Build a ChatOpenAI instance from environment variables.

    Args:
        max_tokens: Override for the maximum number of output tokens.
                    Defaults to the ``MAX_TOKENS`` env var or 16384.
    """
    from langchain_openai import ChatOpenAI

    api_base = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", ""))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("MODEL_NAME", "gpt-4o")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY must be set.  "
            "Also set OPENAI_API_BASE and MODEL_NAME as needed."
        )

    if max_tokens is None:
        max_tokens = int(os.environ.get("MAX_TOKENS", "16384"))
    log.info("Building LLM  model=%s  base_url=%s  max_tokens=%d", model, api_base or "(default)", max_tokens)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=api_base or None,
        temperature=0,
        max_tokens=max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _invoke_and_parse(llm, messages: list[dict], schema: type[BaseModel]):
    """Invoke the LLM, extract JSON from the response, and validate with Pydantic.

    Works around endpoints that don't support ``json_schema`` response_format
    by using ``json_object`` mode and doing client-side validation.
    """
    # Estimate prompt size for the log
    total_chars = sum(len(m.get("content", "")) for m in messages)
    log.info(
        "Sending request to LLM  (prompt ~%s chars across %d messages) …",
        f"{total_chars:,}", len(messages),
    )

    t0 = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - t0

    raw = response.content
    log.info(
        "LLM responded in %.1fs  (response ~%s chars)",
        elapsed, f"{len(raw):,}",
    )

    # Strip markdown fences if the model wraps its answer in ```json ... ```
    text = raw.strip()
    if text.startswith("```"):
        log.debug("Stripping markdown code fences from response")
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    log.info("Parsing JSON response …")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.error("JSON parse failed: %s", exc)
        log.debug("Raw response (first 500 chars): %s", raw[:500])
        raise ValueError(
            f"LLM did not return valid JSON.\n"
            f"JSONDecodeError: {exc}\n"
            f"Raw response (first 500 chars): {raw[:500]}"
        ) from exc

    # The model may return {rules: [...]} or just [...]
    # Normalise bare list → wrapped object for the Pydantic model
    if isinstance(data, list):
        log.info("Response is a bare JSON list (%d items) — wrapping in {rules: [...]}", len(data))
        data = {"rules": data}

    log.info("Validating against %s schema …", schema.__name__)
    try:
        result = schema.model_validate(data)
    except Exception as exc:
        log.error("Schema validation failed: %s", exc)
        raise ValueError(
            f"LLM JSON did not match the {schema.__name__} schema.\n"
            f"Validation error: {exc}\n"
            f"Parsed keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        ) from exc

    log.info("Validation OK  (%d rules parsed)", len(result.rules))
    return result


# ── Document chunking ───────────────────────────────────────────────────────

# Conservative limit for the document portion of each chunk.
# With a 32K-token context window we need room for:
#   - system prompt + few-shot + schema: ~4 K tokens (~14 K chars)
#   - the document chunk itself
#   - model response: up to ~16 K tokens for dense rule-rich sections
# Smaller chunks → fewer rules per chunk → responses stay within max_tokens.
_CHUNK_CHAR_LIMIT = 15_000


def _split_into_chunks(text: str, max_chars: int = _CHUNK_CHAR_LIMIT) -> list[str]:
    """Split a markdown document into chunks that stay under *max_chars*.

    Splitting strategies (tried in order until one produces multiple parts):
      1. Standalone ``Page N`` lines  (e.g. "Page 12")
      2. ``Page N of M`` lines        (e.g. "Page 12 of 261")
      3. Markdown headings             (``# …``, ``## …``, ``### …``)
      4. Paragraph boundaries           (double newlines)
      5. Hard character-limit fallback  (split mid-text every *max_chars*)

    Consecutive parts are grouped into chunks that stay under *max_chars*.
    If a single part exceeds the limit it is returned as its own chunk.
    """

    def _group_parts(parts: list[str], limit: int) -> list[str]:
        """Greedily merge consecutive *parts* into chunks ≤ *limit* chars."""
        chunks: list[str] = []
        current = ""
        for part in parts:
            if current and len(current) + len(part) > limit:
                chunks.append(current)
                current = part
            else:
                current += part
        if current:
            chunks.append(current)
        return chunks

    # Strategy 1: standalone "Page N" markers
    parts = re.split(r"(?=^Page \d+$)", text, flags=re.MULTILINE)
    parts = [p for p in parts if p.strip()]
    if len(parts) > 1:
        log.info("Chunking strategy: standalone 'Page N' markers (%d parts)", len(parts))
        return _group_parts(parts, max_chars)

    # Strategy 2: "Page N of M" markers (common in regulatory PDFs)
    parts = re.split(r"(?=^Page \d+ of \d+$)", text, flags=re.MULTILINE)
    parts = [p for p in parts if p.strip()]
    if len(parts) > 1:
        log.info("Chunking strategy: 'Page N of M' markers (%d parts)", len(parts))
        return _group_parts(parts, max_chars)

    # Strategy 3: Markdown headings (# … / ## … / ### …)
    parts = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    parts = [p for p in parts if p.strip()]
    if len(parts) > 1:
        log.info("Chunking strategy: Markdown headings (%d parts)", len(parts))
        return _group_parts(parts, max_chars)

    # Strategy 4: Paragraph boundaries (double newlines)
    parts = re.split(r"\n\n+", text)
    parts = [p for p in parts if p.strip()]
    if len(parts) > 1:
        log.info("Chunking strategy: paragraph boundaries (%d parts)", len(parts))
        # Re-join with double newlines to preserve formatting when grouping
        parts = [p + "\n\n" for p in parts]
        return _group_parts(parts, max_chars)

    # Strategy 5: Hard character-limit fallback
    log.warning("Chunking strategy: hard character-limit fallback (no structural markers found)")
    chunks: list[str] = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i : i + max_chars])
    return chunks


def _build_extraction_messages(
    chunk_text: str,
    chunk_idx: int,
    total_chunks: int,
    example_json: str,
    schema_hint: str,
) -> list[dict]:
    """Build the prompt messages for a single extraction chunk."""
    chunk_note = ""
    if total_chunks > 1:
        chunk_note = (
            f"\n\nNOTE: This is chunk {chunk_idx}/{total_chunks} of a larger "
            "document.  Extract only the rules present in THIS chunk.  "
            "Use temporary rule_ids (they will be renumbered later)."
        )

    return [
        {
            "role": "system",
            "content": (
                "You are an expert at extracting structured business rules from "
                "regulatory and policy documents.  Given a Markdown document (or a "
                "section of one), identify every distinct business rule, regulatory "
                "requirement, or compliance obligation and return them as a JSON "
                "object that conforms to the provided schema.\n\n"
                "A 'rule' is any statement that prescribes, mandates, restricts, "
                "or conditions behaviour — including numbered regulations (e.g. "
                "R.1.1.25), policy statements, formulas, eligibility criteria, "
                "disclosure requirements, prohibitions, or procedural obligations.\n\n"
                "Guidelines:\n"
                "- Assign sequential rule_ids starting from R001.\n"
                "- Infer the entity_applied from context — this is the primary "
                "entity the rule governs (e.g. 'school', 'student', 'bank', "
                "'credit_institution', 'regulated_person', 'client', 'lender', "
                "'district', etc.).\n\n"
                "IMPORTANT — data_required format:\n"
                "- data_required is a list of objects, each with:\n"
                "  - data_source: the system or data domain name. If the document "
                "names specific systems (e.g. 'Entity Profile System (EPS)'), use "
                "those. Otherwise, infer a logical data domain from context (e.g. "
                "'Client Records (CR)', 'Loan Management System (LMS)', "
                "'Compliance Registry (COMP)').\n"
                "  - data_attributes: list of {attribute_name, description, "
                "example_values, data_type}\n"
                "- data_type must be one of: string, integer, float, boolean, date.\n"
                "  Use integer for counts; float for rates/percentages; boolean "
                "for flags; date for dates; string for IDs, codes, and other text.\n"
                "- Infer the specific attributes (fields) needed based on the "
                "business rules, conditions, and formulas described.\n"
                "- Use snake_case for attribute_name (e.g. 'is_public_school', "
                "'loan_amount', 'interest_rate', 'client_type').\n"
                "- Provide 2-4 realistic example_values for each attribute.\n\n"
                "IMPORTANT — conditions and action format:\n"
                "- Every condition MUST reference the data source abbreviation "
                "and attribute name in dot notation, e.g.:\n"
                "  'EPS.is_public_school == true'\n"
                "  'LMS.loan_amount > 50000'\n"
                "  'CR.client_type == \"retail\"'\n"
                "- The action MUST reference the exact data_source.attribute used.\n"
                "- This makes conditions machine-executable against the data.\n\n"
                "Other guidelines:\n"
                "- Extract every numbered rule, bullet, or sub-bullet that "
                "prescribes a requirement as a separate rule.\n"
                "- Derive the action and outcome from the rule text and its "
                "surrounding context (definitions, sections, headings).\n"
                "- Leave outgoing_routes empty for now (routes are discovered in "
                "a later step); set is_final=false.\n"
                "- When information is implied by context but not explicitly "
                "stated, include it as a condition.\n"
                "- If a chunk truly contains NO extractable rules (e.g. it is "
                "only a table of contents or glossary with no obligations), "
                "return {\"rules\": []}.\n\n"
                "You MUST respond with valid JSON only (no markdown fences, no "
                "commentary) matching this schema:\n"
                f"{schema_hint}"
                f"{chunk_note}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is an example.\n\n"
                "--- SOURCE TEXT ---\n"
                f"{_EXAMPLE_SOURCE}\n"
                "--- EXPECTED OUTPUT ---\n"
                f"{example_json}"
            ),
        },
        {
            "role": "assistant",
            "content": example_json,
        },
        {
            "role": "user",
            "content": (
                "Now extract ALL business rules from the following document section.  "
                "Return them in the same JSON schema.  Respond with JSON only.\n\n"
                "--- DOCUMENT ---\n"
                f"{chunk_text}"
            ),
        },
    ]


def create_rules(
    input_md: str,
    output_json: str = "rules.json",
) -> str:
    """Extract business rules from a Markdown file using LangChain structured extraction.

    If the document exceeds the model's context window, it is automatically
    split into chunks on ``Page N`` boundaries. Each chunk is processed
    independently and the results are merged and renumbered sequentially.

    Args:
        input_md:    Path to the Markdown file produced by ``convert_pdf_to_markdown``.
        output_json: Path where the extracted rules JSON will be saved.

    Returns:
        The path to the generated JSON file.
    """
    log.info("=" * 60)
    log.info("CREATE RULES  input=%s  output=%s", input_md, output_json)
    log.info("=" * 60)

    md_path = Path(input_md)
    if not md_path.exists():
        raise FileNotFoundError(f"Input markdown file not found: {input_md}")

    md_text = md_path.read_text(encoding="utf-8")
    log.info("Read markdown file: %s chars", f"{len(md_text):,}")

    llm = _build_extraction_llm()

    # Prepare reusable prompt fragments
    example_json = _EXAMPLE_OUTPUT.model_dump_json(indent=2)
    schema_hint = json.dumps(RuleSet.model_json_schema(), indent=2)

    # Split into chunks if the document is large
    chunks = _split_into_chunks(md_text)
    log.info(
        "Document split into %d chunk(s)  (limit=%s chars/chunk)",
        len(chunks), f"{_CHUNK_CHAR_LIMIT:,}",
    )
    for i, c in enumerate(chunks, 1):
        log.info("  Chunk %d: %s chars", i, f"{len(c):,}")

    # Process each chunk
    all_rules: list[Rule] = []
    for idx, chunk in enumerate(chunks, 1):
        log.info("-" * 40)
        log.info("Processing chunk %d/%d  (%s chars) …",
                 idx, len(chunks), f"{len(chunk):,}")

        messages = _build_extraction_messages(
            chunk, idx, len(chunks), example_json, schema_hint,
        )

        log.info("Invoking LLM for rule extraction (chunk %d/%d) …", idx, len(chunks))
        result: RuleSet = _invoke_and_parse(llm, messages, RuleSet)
        log.info("Chunk %d/%d yielded %d rules", idx, len(chunks), len(result.rules))

        for rule in result.rules:
            rule.chunk_id = idx
        all_rules.extend(result.rules)

    # Renumber rule_ids sequentially across all chunks
    log.info("-" * 40)
    log.info("Merging & renumbering %d total rules …", len(all_rules))
    for i, rule in enumerate(all_rules, 1):
        rule.rule_id = f"R{i:03d}"

    # Serialise to JSON (flat list to match rules.json format)
    rules_dicts = [r.model_dump() for r in all_rules]
    output_path = Path(output_json)
    output_path.write_text(
        json.dumps(rules_dicts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d rules to '%s'", len(rules_dicts), output_json)
    for r in all_rules:
        log.info("  %s  [chunk %s]  %s  (entity=%s, %d conditions)",
                 r.rule_id, r.chunk_id, r.rule_name, r.entity_applied, len(r.conditions))
    return str(output_path)


# ── Route discovery ─────────────────────────────────────────────────────────

_ROUTE_EXAMPLE_INPUT = [
    {
        "rule_id": "R001",
        "rule_name": "Grades Served",
        "entity_applied": "school",
        "data_required": [
            {"data_source": "Entity Profile System (EPS)", "data_attributes": [
                {"attribute_name": "school_id", "description": "Unique school ID", "example_values": ["SCH00012345"], "data_type": "string"},
                {"attribute_name": "is_public_school", "description": "Public school flag", "example_values": ["true", "false"], "data_type": "boolean"},
                {"attribute_name": "grades_offered", "description": "Grades offered", "example_values": ["PK-12", "K-8"], "data_type": "string"},
            ]},
            {"data_source": "Student Information System (SIS)", "data_attributes": [
                {"attribute_name": "student_id", "description": "Student ID", "example_values": ["S0000012345"], "data_type": "string"},
                {"attribute_name": "school_id", "description": "School ID", "example_values": ["SCH00012345"], "data_type": "string"},
                {"attribute_name": "grade_level", "description": "Grade level", "example_values": ["PK", "K", "12"], "data_type": "string"},
            ]},
        ],
        "conditions": [
            "EPS.is_public_school == 'true' (school must be a public school)",
            "Use EPS.grades_offered to determine grades served",
        ],
        "action": "Retrieve EPS.grades_offered for the school identified by EPS.school_id.",
        "outcome": "Grade levels served by the school (e.g., PK–12).",
        "outgoing_routes": [],
        "is_final": False,
    },
    {
        "rule_id": "R002",
        "rule_name": "Report Card Eligible Entity",
        "entity_applied": "school",
        "data_required": [
            {"data_source": "Entity Profile System (EPS)", "data_attributes": [
                {"attribute_name": "school_id", "description": "School ID", "example_values": ["SCH00012345"], "data_type": "string"},
                {"attribute_name": "school_category", "description": "Category code", "example_values": ["4", "8"], "data_type": "integer"},
                {"attribute_name": "RCDTS_code", "description": "RCDTS identifier", "example_values": ["01001000102001", "3000", "9000"], "data_type": "string"},
                {"attribute_name": "is_public_school", "description": "Public school flag", "example_values": ["true", "false"], "data_type": "boolean"},
            ]},
            {"data_source": "Student Information System (SIS)", "data_attributes": [
                {"attribute_name": "home_school_id", "description": "Home school ID", "example_values": ["SCH00012345"], "data_type": "string"},
                {"attribute_name": "serving_school_id", "description": "Serving school ID", "example_values": ["SCH00012345"], "data_type": "string"},
                {"attribute_name": "home_enrollment_count", "description": "Enrollment at home school", "example_values": ["10", "25", "150"], "data_type": "integer"},
                {"attribute_name": "serving_enrollment_count", "description": "Enrollment at serving school", "example_values": ["10", "50", "200"], "data_type": "integer"},
            ]},
        ],
        "conditions": [
            "EPS.school_category in ('4', '8') (public school Category 4 or Category 8)",
            "SIS.home_enrollment_count >= 10 OR SIS.serving_enrollment_count >= 10",
        ],
        "action": "Check EPS.school_category and SIS enrollment counts to determine eligibility.",
        "outcome": "Binary flag: entity is eligible (Yes/No) to receive a Report Card.",
        "outgoing_routes": [],
        "is_final": False,
    },
]

_ROUTE_EXAMPLE_OUTPUT = RoutedRuleSet(rules=[
    RoutedRule(
        rule_id="R001",
        rule_name="Grades Served",
        entity_applied="school",
        data_required=[
            DataRequirement(data_source="Entity Profile System (EPS)", data_attributes=[
                DataAttribute(attribute_name="school_id", description="Unique school ID", example_values=["SCH00012345"], data_type="string"),
                DataAttribute(attribute_name="is_public_school", description="Public school flag", example_values=["true", "false"], data_type="boolean"),
                DataAttribute(attribute_name="grades_offered", description="Grades offered", example_values=["PK-12", "K-8"], data_type="string"),
            ]),
            DataRequirement(data_source="Student Information System (SIS)", data_attributes=[
                DataAttribute(attribute_name="student_id", description="Student ID", example_values=["S0000012345"], data_type="string"),
                DataAttribute(attribute_name="school_id", description="School ID", example_values=["SCH00012345"], data_type="string"),
                DataAttribute(attribute_name="grade_level", description="Grade level", example_values=["PK", "K", "12"], data_type="string"),
            ]),
        ],
        conditions=[
            "EPS.is_public_school == 'true' (school must be a public school)",
            "Use EPS.grades_offered to determine grades served",
        ],
        action="Retrieve EPS.grades_offered for the school identified by EPS.school_id.",
        outcome="Grade levels served by the school (e.g., PK–12).",
        outgoing_routes=[
            OutgoingRoute(condition="Grades determined", next_rule="R002"),
        ],
        is_final=False,
    ),
    RoutedRule(
        rule_id="R002",
        rule_name="Report Card Eligible Entity",
        entity_applied="school",
        data_required=[
            DataRequirement(data_source="Entity Profile System (EPS)", data_attributes=[
                DataAttribute(attribute_name="school_id", description="School ID", example_values=["SCH00012345"], data_type="string"),
                DataAttribute(attribute_name="school_category", description="Category code", example_values=["4", "8"], data_type="integer"),
                DataAttribute(attribute_name="RCDTS_code", description="RCDTS identifier", example_values=["01001000102001", "3000", "9000"], data_type="string"),
                DataAttribute(attribute_name="is_public_school", description="Public school flag", example_values=["true", "false"], data_type="boolean"),
            ]),
            DataRequirement(data_source="Student Information System (SIS)", data_attributes=[
                DataAttribute(attribute_name="home_school_id", description="Home school ID", example_values=["SCH00012345"], data_type="string"),
                DataAttribute(attribute_name="serving_school_id", description="Serving school ID", example_values=["SCH00012345"], data_type="string"),
                DataAttribute(attribute_name="home_enrollment_count", description="Enrollment at home school", example_values=["10", "25", "150"], data_type="integer"),
                DataAttribute(attribute_name="serving_enrollment_count", description="Enrollment at serving school", example_values=["10", "50", "200"], data_type="integer"),
            ]),
        ],
        conditions=[
            "EPS.school_category in ('4', '8') (public school Category 4 or Category 8)",
            "SIS.home_enrollment_count >= 10 OR SIS.serving_enrollment_count >= 10",
        ],
        action="Check EPS.school_category and SIS enrollment counts to determine eligibility.",
        outcome="Binary flag: entity is eligible (Yes/No) to receive a Report Card.",
        outgoing_routes=[
            OutgoingRoute(condition="School is eligible", next_rule="EXTERNAL_MISSING"),
            OutgoingRoute(condition="School is NOT eligible", next_rule=None),
        ],
        is_final=False,
    ),
])


def _build_rule_directory(
    rules: list[dict],
    max_chars: int = 30_000,
) -> str:
    """Build a compact one-line-per-rule directory for route-target lookup.

    This is included in every routing chunk so the model can reference any
    rule_id as a potential target, even rules processed in other chunks.

    If the full directory exceeds *max_chars*, entries are progressively
    shortened until they fit.
    """
    # Try detailed format first, then progressively shorter
    formats = [
        lambda r: (
            f"{r['rule_id']}: {r['rule_name']} "
            f"(entity={r['entity_applied']}) — "
            f"action: {r['action'][:80]}… | "
            f"outcome: {r['outcome'][:80]}…"
        ),
        lambda r: (
            f"{r['rule_id']}: {r['rule_name']} "
            f"(entity={r['entity_applied']})"
        ),
        lambda r: f"{r['rule_id']}: {r['rule_name'][:50]}",
    ]

    for fmt in formats:
        directory = "\n".join(fmt(r) for r in rules)
        if len(directory) <= max_chars:
            return directory

    # Last resort: just rule_ids
    return "\n".join(r["rule_id"] for r in rules)


def _chunk_rules_list(
    rules: list[dict],
    max_chars: int = _CHUNK_CHAR_LIMIT,
    max_rules: int | None = None,
) -> list[list[dict]]:
    """Split a list of rule dicts into batches that stay under *max_chars*
    when serialised to JSON, and optionally under *max_rules* per batch."""
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_size = 0

    for rule in rules:
        rule_json = json.dumps(rule, ensure_ascii=False)
        rule_size = len(rule_json)

        hit_char_limit = current_batch and current_size + rule_size > max_chars
        hit_rule_limit = max_rules and len(current_batch) >= max_rules

        if current_batch and (hit_char_limit or hit_rule_limit):
            batches.append(current_batch)
            current_batch = [rule]
            current_size = rule_size
        else:
            current_batch.append(rule)
            current_size += rule_size

    if current_batch:
        batches.append(current_batch)
    return batches


def _try_resolve_invalid_route(
    route: dict,
    rule_dict: dict,
    valid_rule_ids: set[str],
    doc_chunks: list[str],
    rule_directory: str,
    llm,
) -> str | None:
    """Try to resolve an invalid next_rule (e.g. EXTERNAL_MISSING) using doc chunks.

    Loops over each doc chunk and asks the LLM to find a matching rule_id.
    Returns the resolved rule_id if found, None otherwise.
    """
    rule_id = rule_dict.get("rule_id", "")
    rule_name = rule_dict.get("rule_name", "")
    action = rule_dict.get("action", "")[:200]
    outcome = rule_dict.get("outcome", "")[:200]
    condition = route.get("condition", "")
    invalid_target = route.get("next_rule", "EXTERNAL_MISSING")

    valid_list = sorted(valid_rule_ids)
    for chunk_idx, chunk in enumerate(doc_chunks, 1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at matching business rules to their logical successors.\n\n"
                    "A rule has an outgoing route that points to an invalid/missing target "
                    f"({invalid_target}). Given the source rule's action/outcome and a "
                    "document chunk, find which rule_id from the VALID LIST best represents "
                    "the intended next step.\n\n"
                    "Respond with JSON only: {\"rule_id\": \"Rxxx\"} or {\"rule_id\": null} "
                    "if no valid rule matches. Use null when the document describes something "
                    "outside the rule set."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"SOURCE RULE: {rule_id} - {rule_name}\n"
                    f"Action: {action}\nOutcome: {outcome}\n"
                    f"Route condition: {condition}\n"
                    f"Invalid target: {invalid_target}\n\n"
                    f"VALID RULE IDs: {', '.join(valid_list)}\n\n"
                    f"RULE DIRECTORY (for semantic matching):\n{rule_directory}\n\n"
                    f"--- DOCUMENT CHUNK {chunk_idx}/{len(doc_chunks)} ---\n{chunk[:15000]}"
                ),
            },
        ]
        try:
            response = llm.invoke(messages)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            data = json.loads(text)
            resolved = data.get("rule_id")
            if resolved is not None and resolved in valid_rule_ids:
                log.info(
                    "  Resolved %s route %r -> %s (via doc chunk %d)",
                    rule_id, condition[:50], resolved, chunk_idx,
                )
                return resolved
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.debug("  Chunk %d: LLM resolution failed: %s", chunk_idx, e)
            continue
    return None


def _detect_and_break_loops(routed_dicts: list[dict]) -> list[dict]:
    """Detect and break routing loops introduced by the LLM.

    Detects three patterns:
    1. **Self-loops**: A rule routes to itself (next_rule == own rule_id).
    2. **Two-node cycles**: A -> B -> A (ping-pong between two rules).
    3. **Unconditional routing**: routing_cel is a bare string literal
       like ``'R200'`` with no exit condition.

    Offending routes are removed. If a rule loses all routes, it becomes
    ``is_final=True``.
    """
    rule_map = {r["rule_id"]: r for r in routed_dicts}
    loops_broken = 0

    for rule_dict in routed_dicts:
        rid = rule_dict["rule_id"]
        routes = list(rule_dict.get("outgoing_routes") or [])
        cleaned_routes = []
        for route in routes:
            next_rule = route.get("next_rule")
            if next_rule is None:
                cleaned_routes.append(route)
                continue

            # Pattern 1: Self-loop — rule routes back to itself
            if next_rule == rid:
                log.warning(
                    "  LOOP DETECTED: %s routes to itself (condition: %r) — removing route",
                    rid, route.get("condition", "")[:50],
                )
                loops_broken += 1
                continue

            # Pattern 2: Two-node cycle — A -> B -> A
            target = rule_map.get(next_rule)
            if target:
                target_routes = target.get("outgoing_routes") or []
                for tr in target_routes:
                    if tr.get("next_rule") == rid:
                        log.warning(
                            "  LOOP DETECTED: %s -> %s -> %s (2-node cycle) — "
                            "removing route from %s back to %s",
                            rid, next_rule, rid, next_rule, rid,
                        )
                        # Remove the back-edge from the target rule
                        target["outgoing_routes"] = [
                            r for r in target_routes
                            if r.get("next_rule") != rid
                        ]
                        if not target["outgoing_routes"]:
                            target["is_final"] = True
                            log.info("  %s: No routes left after cycle break; marked is_final=True", next_rule)
                        loops_broken += 1
                        break

            cleaned_routes.append(route)

        rule_dict["outgoing_routes"] = cleaned_routes
        if not cleaned_routes:
            rule_dict["is_final"] = True
            log.info("  %s: No routes left after loop removal; marked is_final=True", rid)

    if loops_broken:
        log.info("Loop detection: broke %d loop(s)", loops_broken)
    else:
        log.info("Loop detection: no loops found")

    return routed_dicts


def _verify_and_fix_routes(
    routed_dicts: list[dict],
    input_json: str,
    llm,
) -> list[dict]:
    """Verify all next_rule targets exist; fix or remove invalid routes.

    For each route where next_rule is not in the rule set:
    - Try to resolve using LLM + doc chunks (loop until resolved or exhausted)
    - If unresolved: remove the route
    - If rule has no routes left: set is_final=True

    After validation, runs loop detection to break self-loops and cycles.
    """
    valid_rule_ids = {r["rule_id"] for r in routed_dicts}
    rule_directory = _build_rule_directory(routed_dicts)

    md_path = Path(input_json).with_suffix(".md")
    doc_chunks: list[str] = []
    if md_path.exists():
        doc_chunks = _split_into_chunks(md_path.read_text(encoding="utf-8"))
        log.info("Load doc chunks for route validation: %d chunks from '%s'", len(doc_chunks), md_path)
    else:
        log.info("No markdown doc at '%s'; will remove invalid routes without resolution", md_path)

    fixed = []
    for rule_dict in routed_dicts:
        routes = list(rule_dict.get("outgoing_routes") or [])
        valid_routes = []
        for route in routes:
            next_rule = route.get("next_rule")
            if next_rule is None:
                valid_routes.append(route)
            elif next_rule in valid_rule_ids:
                valid_routes.append(route)
            else:
                resolved = _try_resolve_invalid_route(
                    route, rule_dict, valid_rule_ids, doc_chunks,
                    rule_directory, llm,
                )
                if resolved is not None:
                    valid_routes.append({**route, "next_rule": resolved})
                else:
                    log.warning(
                        "  %s: Removing invalid route %r -> %s (could not resolve)",
                        rule_dict["rule_id"], route.get("condition", "")[:40], next_rule,
                    )
        rule_dict["outgoing_routes"] = valid_routes
        if not valid_routes:
            rule_dict["is_final"] = True
            log.info("  %s: No routes left; marked is_final=True", rule_dict["rule_id"])
        fixed.append(rule_dict)

    # Post-validation: detect and break routing loops
    log.info("-" * 40)
    log.info("Detecting and breaking routing loops …")
    fixed = _detect_and_break_loops(fixed)

    return fixed


def create_routes(
    input_json: str,
    output_json: str | None = None,
) -> str:
    """Discover logical routes between extracted business rules using an LLM.

    If the rules set is too large for a single request, it is processed in
    batches.  A compact directory of ALL rules is included in every request
    so the model can reference any rule_id as a route target.

    Args:
        input_json:  Path to the rules JSON (flat list of rule dicts).
        output_json: Path for the routed output.  Defaults to
                     ``<stem>_routed.json``.

    Returns:
        The path to the generated routed-rules JSON file.
    """
    log.info("=" * 60)
    log.info("CREATE ROUTES  input=%s  output=%s", input_json, output_json)
    log.info("=" * 60)

    in_path = Path(input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input rules file not found: {input_json}")

    if output_json is None:
        output_json = str(in_path.with_stem(in_path.stem + "_routed"))

    all_rules: list[dict] = json.loads(in_path.read_text(encoding="utf-8"))
    log.info("Read %d rules from '%s'", len(all_rules), input_json)

    # Use a smaller max_tokens for routing — the output is the batch rules
    # with routes added, which is more compact than rule extraction.
    llm = _build_extraction_llm(max_tokens=8192)

    # Build a compact directory of every rule (included in every chunk)
    directory = _build_rule_directory(all_rules)
    log.info("Rule directory: %s chars  (%d rules)", f"{len(directory):,}", len(all_rules))

    # Prepare reusable prompt fragments
    example_input_json = json.dumps(_ROUTE_EXAMPLE_INPUT, indent=2)
    example_output_json = _ROUTE_EXAMPLE_OUTPUT.model_dump_json(indent=2)
    schema_hint = json.dumps(RoutedRuleSet.model_json_schema(), indent=2)

    # Chunk the rules list — measure actual overhead to stay within context.
    # Build a dummy prompt (with empty batch) to measure fixed overhead.
    _dummy_system = (
        schema_hint + "\n" +
        "You are an expert at analysing business-rule dependencies.\n"
    )
    _fixed_overhead = (
        len(_dummy_system)
        + len(example_input_json)
        + len(example_output_json) * 2  # appears twice (user + assistant)
        + len(directory)
        + 2_000  # instruction text, labels, whitespace
    )
    # Context budget: 32K tokens ≈ 128K chars.
    # Reserve 8K tokens (≈32K chars) for output + safety margin.
    max_input_chars = 80_000  # ~20K tokens input (leaves 12K for output + margin)
    available_for_batch = max(max_input_chars - _fixed_overhead, 6_000)
    log.info("Routing batch budget: %s chars available for batch "
             "(fixed overhead=%s, directory=%s)",
             f"{available_for_batch:,}", f"{_fixed_overhead:,}", f"{len(directory):,}")
    max_rules_per_batch = 20
    batches = _chunk_rules_list(all_rules, max_chars=available_for_batch, max_rules=max_rules_per_batch)
    log.info(
        "Rules split into %d batch(es)  (batch limit=%s chars, max %d rules/batch)",
        len(batches), f"{available_for_batch:,}", max_rules_per_batch,
    )
    for i, b in enumerate(batches, 1):
        batch_json = json.dumps(b, ensure_ascii=False)
        log.info("  Batch %d: %d rules, %s chars", i, len(b), f"{len(batch_json):,}")

    # Process each batch
    all_routed: list[RoutedRule] = []
    for idx, batch in enumerate(batches, 1):
        log.info("-" * 40)
        log.info("Processing batch %d/%d  (%d rules) …", idx, len(batches), len(batch))

        batch_json = json.dumps(batch, indent=2, ensure_ascii=False)
        batch_note = ""
        if len(batches) > 1:
            batch_note = (
                f"\n\nNOTE: This is batch {idx}/{len(batches)}.  "
                "Discover routes ONLY for the rules in the BATCH below.  "
                "Use the DIRECTORY to find valid next_rule targets across "
                "the full rule set."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at analysing business-rule dependencies.\n\n"
                    "I have a list of extracted Business Rules, each with a unique ID. "
                    "Your task is to identify logical relationships and \"handoffs\" "
                    "between these rules.\n\n"
                    "## INSTRUCTIONS\n"
                    "1. Analyse the \"action\" and \"outcome\" of every rule in the BATCH.\n"
                    "2. Look for other rules (in the BATCH or the DIRECTORY) whose "
                    "\"rule_name\", \"conditions\", or \"entity_applied\" logically "
                    "follows that action/outcome.\n"
                    "3. Be semantically smart: \"Forward to Risk\" (Rule A) matches "
                    "\"Risk Assessment Team\" (Rule B).  Outcomes that produce data "
                    "consumed by another rule's conditions create a route.\n"
                    "4. If a rule implies a next step but the target rule is missing "
                    "from both the BATCH and the DIRECTORY, set next_rule to "
                    "\"EXTERNAL_MISSING\".\n"
                    "5. If a rule has NO outgoing routes at all, set is_final to true; "
                    "otherwise set is_final to false.\n"
                    "6. Preserve every other field of each rule unchanged.\n"
                    "7. Return ONLY the rules from the BATCH (with routes populated), "
                    "NOT the entire directory.\n\n"
                    "## OUTPUT\n"
                    "Return the batch rules with their outgoing_routes and "
                    "is_final fields populated.\n\n"
                    "You MUST respond with valid JSON only (no markdown fences, no "
                    "commentary) matching this schema:\n"
                    f"{schema_hint}"
                    f"{batch_note}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is an example.\n\n"
                    "--- INPUT RULES ---\n"
                    f"{example_input_json}\n\n"
                    "--- EXPECTED OUTPUT ---\n"
                    f"{example_output_json}"
                ),
            },
            {
                "role": "assistant",
                "content": example_output_json,
            },
            {
                "role": "user",
                "content": (
                    "Now discover routes for the BATCH of rules below.  "
                    "Use the DIRECTORY to identify valid next_rule targets.  "
                    "Respond with JSON only.\n\n"
                    "--- FULL RULE DIRECTORY (for target lookup) ---\n"
                    f"{directory}\n\n"
                    "--- BATCH OF RULES TO PROCESS ---\n"
                    f"{batch_json}"
                ),
            },
        ]

        log.info("Invoking LLM for route discovery (batch %d/%d) …", idx, len(batches))
        result: RoutedRuleSet = _invoke_and_parse(llm, messages, RoutedRuleSet)
        log.info("Batch %d/%d: %d rules routed", idx, len(batches), len(result.rules))

        all_routed.extend(result.rules)

    # Write merged results
    log.info("-" * 40)
    log.info("Merging %d routed rules …", len(all_routed))
    routed_dicts = [r.model_dump() for r in all_routed]

    # Step 4: Verify all routes point to existing rules; fix or remove invalid ones
    log.info("-" * 40)
    log.info("Verify and fix invalid routes …")
    routed_dicts = _verify_and_fix_routes(routed_dicts, str(in_path), llm)
    out_path = Path(output_json)
    out_path.write_text(
        json.dumps(routed_dicts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d routed rules to '%s'", len(routed_dicts), output_json)
    total_routes = sum(len(r.get("outgoing_routes") or []) for r in routed_dicts)
    final_count = sum(1 for r in routed_dicts if r.get("is_final"))
    log.info("  Total outgoing routes: %d", total_routes)
    log.info("  Terminal (is_final) rules: %d", final_count)
    for r in routed_dicts:
        routes = r.get("outgoing_routes") or []
        if routes:
            targets = ", ".join(
                f"{rt.get('condition', '')} -> {rt.get('next_rule') or 'END'}"
                for rt in routes
            )
            log.info("  %s  %s  routes: [%s]", r["rule_id"], r["rule_name"], targets)
        else:
            log.info("  %s  %s  (terminal)", r["rule_id"], r["rule_name"])
    return str(out_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__.strip())
        sys.exit(1)

    input_pdf = sys.argv[1]
    pdf_path = Path(input_pdf)

    output_md = str(pdf_path.with_suffix(".md"))
    output_json = str(pdf_path.with_suffix(".json"))
    output_routed = str(pdf_path.with_stem(pdf_path.stem + "_routed").with_suffix(".json"))

    # Step 1 – PDF → Markdown
    try:
        convert_pdf_to_markdown(input_pdf, output_md)
        print(f"[1/3] Converted '{input_pdf}' -> '{output_md}'")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2 – Markdown → rules JSON
    try:
        create_rules(output_md, output_json)
        print(f"[2/3] Extracted rules '{output_md}' -> '{output_json}'")
    except Exception as e:
        print(f"Rule extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3 – Discover routes between rules
    try:
        create_routes(output_json, output_routed)
        print(f"[3/3] Discovered routes '{output_json}' -> '{output_routed}'")
    except Exception as e:
        print(f"Route discovery failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
