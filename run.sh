#!/bin/sh
set -e

# ─── Full pipeline: 01_convert_pdf_to_rules → 06_review_report ──────────────
#
# Usage:
#   ./run.sh <path/to/input.pdf> [--num-rows N]
#
# Examples:
#   ./run.sh use_case_01/Public-Business-Rules-2024-Report-Card-Metrics.pdf
#   ./run.sh use_case_02/Conduct-of-Business-Rulebook.pdf --num-rows 20
#
# The PDF path determines the output directory and filename stem.
# All intermediate files are written alongside the PDF.
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$1" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  echo "Usage: $0 <path/to/input.pdf> [--num-rows N]"
  echo ""
  echo "Runs the full rule-engine pipeline (steps 01–06) on the given PDF."
  echo ""
  echo "Options:"
  echo "  --num-rows N   Number of mock-data rows per rule (default: 50)"
  exit 1
fi

INPUT_PDF="$1"
shift

# Default number of mock-data rows
NUM_ROWS=50

# Parse remaining arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --num-rows)
      NUM_ROWS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Validate input file
if [ ! -f "$INPUT_PDF" ]; then
  echo "Error: File not found: $INPUT_PDF"
  exit 1
fi

# Derive the stem (path without .pdf extension) for intermediate filenames
STEM="${INPUT_PDF%.pdf}"

echo "============================================================"
echo "  PIPELINE: $INPUT_PDF"
echo "  Mock-data rows per rule: $NUM_ROWS"
echo "============================================================"
echo ""

# ── Step 1: PDF → Markdown → Rules JSON → Routed Rules JSON ─────────────
echo ">>> Step 1/6: Convert PDF to rules (01_convert_pdf_to_rules.py)"
python 01_convert_pdf_to_rules.py "$INPUT_PDF"
echo ""

# ── Step 2: Build graph from routed rules ────────────────────────────────
ROUTED="${STEM}_routed.json"
echo ">>> Step 2/6: Build graph (02_build_graph.py)"
python 02_build_graph.py "$ROUTED"
echo ""

# ── Step 3: Compile CEL expressions ─────────────────────────────────────
echo ">>> Step 3/6: Generate CEL expressions (03_generate_cel.py)"
python 03_generate_cel.py "$ROUTED"
echo ""

# ── Step 4: Generate mock data ──────────────────────────────────────────
CEL="${STEM}_routed_cel.json"
echo ">>> Step 4/6: Generate mock data (04_generate_data.py)"
python 04_generate_data.py "$CEL" --num-rows "$NUM_ROWS"
echo ""

# ── Step 5: Execute rules ───────────────────────────────────────────────
MOCKDATA="${STEM}_routed_cel_mockdata.json"
STARTING_ORDER="${STEM}_routed_starting_order.json"

echo ">>> Step 5/6: Execute rules (05_execute_rules.py)"
python 05_execute_rules.py "$CEL" "$MOCKDATA" \
  --starting-order "$STARTING_ORDER" \
  --num-rows 5

# Find the most recently created execution report for this stem
EXEC_REPORT=$(ls -t "${STEM}_routed_execution_"*.json 2>/dev/null | head -1)
if [ -z "$EXEC_REPORT" ]; then
  echo "Error: No execution report found for stem: $STEM"
  exit 1
fi
echo "  Execution report: $EXEC_REPORT"
echo ""

# ── Step 6: Review report ───────────────────────────────────────────────
echo ">>> Step 6/6: Review report (06_review_report.py)"
python 06_review_report.py "$EXEC_REPORT" --rules-file "$CEL"
echo ""

echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
