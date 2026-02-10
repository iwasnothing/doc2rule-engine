"""
Generate mock data for any data source defined in data_schema.json.

The ``get_data`` function accepts a data-item key (or a substring /
case-insensitive match) and returns a dict whose keys are the schema
fields and whose values are realistic fakes produced by the Faker library.

Usage:
    pip install faker
    python 03_generate_data.py                        # demo all items
    python 03_generate_data.py "SIS EL status table"  # one item
    python 03_generate_data.py "SIS EL status table" 5  # 5 rows
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(0)
random.seed(0)

# ── Load schema once at module level ──────────────────────────────────────────
_SCHEMA_PATH = Path(__file__).parent / "data_schema.json"

def _load_schema() -> dict:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)

_SCHEMA: dict = _load_schema()


# ── Field-name → Faker generator mapping ─────────────────────────────────────
# Each entry is a *callable* that returns a single fake value.
# We match field names with simple substring / regex checks so the
# mapping stays maintainable even as new fields are added.

def _rand_id(prefix: str = "", length: int = 8) -> str:
    digits = "".join(str(random.randint(0, 9)) for _ in range(length))
    return f"{prefix}{digits}"

def _rand_rcdts() -> str:
    """15-digit RCDTS code: 2-region, 3-county, 4-district, 2-type, 4-school."""
    return (
        f"{random.randint(1,99):02d}"
        f"{random.randint(1,102):03d}"
        f"{random.randint(1,9999):04d}"
        f"{random.choice(['02','04','08']):s}"
        f"{random.randint(1,9999):04d}"
    )

def _rand_school_year() -> str:
    y = random.randint(2020, 2024)
    return f"{y}-{str(y+1)[-2:]}"

def _rand_grade() -> str:
    return random.choice(
        ["PK", "K"] + [str(g) for g in range(1, 13)]
    )

def _rand_exit_code() -> str:
    return f"{random.choice([2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20]):02d}"

def _rand_letter_grade() -> str:
    return random.choice([
        "A+","A","A-","B+","B","B-","C+","C","C-",
        "D+","D","D-","F","S","P","U","I","W","WP",
    ])

def _rand_subject() -> str:
    return random.choice(["ELA", "Math", "Science", "Social Studies"])

def _rand_race() -> str:
    return random.choice([
        "White", "Black or African American", "Hispanic or Latino",
        "Asian", "American Indian or Alaska Native",
        "Native Hawaiian or Other Pacific Islander", "Two or More Races",
    ])

def _rand_gender() -> str:
    return random.choice(["Male", "Female", "Non-Binary"])

def _rand_flag() -> str:
    return random.choice(["Y", "N"])

def _rand_bool() -> bool:
    return random.choice([True, False])

def _rand_performance_level(max_level: int = 5) -> int:
    return random.randint(1, max_level)

def _rand_scale_score(low: int = 200, high: int = 800) -> int:
    return random.randint(low, high)

def _rand_percent() -> float:
    return round(random.uniform(0, 100), 2)

def _rand_dollar() -> float:
    return round(random.uniform(1000, 5_000_000), 2)

def _rand_fte() -> float:
    return round(random.choice([0.25, 0.5, 0.75, 1.0]), 2)

def _rand_count(low: int = 0, high: int = 500) -> int:
    return random.randint(low, high)

def _rand_date(start_year: int = 2020, end_year: int = 2025) -> str:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()

def _rand_year() -> int:
    return random.randint(2020, 2025)

def _rand_cip_code() -> str:
    return f"{random.randint(1,52):02d}.{random.randint(0,9999):04d}"

def _rand_endorsement_code() -> str:
    return random.choice([
        "DA", "THDR", "VARS", "MUAP", "BAND", "ORCH",
        "CHOR", "MUGS", "ARTS", "ENG", "MATH", "SCI",
    ])

def _rand_course_id() -> str:
    return f"{random.randint(1,52):02d}{random.randint(1,999):03d}A{random.randint(0,9):03d}"

def _rand_ein() -> str:
    return _rand_id("E", 9)

def _rand_tier() -> str:
    return random.choice(["Tier 1", "Tier 2", "Tier 3", "Tier 4"])

def _rand_designation() -> str:
    return random.choice([
        "Commendable", "Exemplary", "Targeted",
        "Comprehensive", "No Designation",
    ])

def _rand_status_code() -> str:
    return random.choice(["F", "H", "Q", "W", "A", "D", "L", "G"])

def _rand_institution_level() -> str:
    return random.choice(["2", "4", "L"])

def _rand_institution_type() -> str:
    return random.choice(["Public", "Private"])


# ── Pattern-based generator resolution ────────────────────────────────────────
# Order matters: first match wins.

_FIELD_GENERATORS: list[tuple[re.Pattern, callable]] = [
    # ── IDs ───────────────────────────────────────────────────────────────
    (re.compile(r"student_id"),              lambda: _rand_id("S", 10)),
    (re.compile(r"enrollment_id"),           lambda: _rand_id("EN", 8)),
    (re.compile(r"score_id"),                lambda: _rand_id("SC", 8)),
    (re.compile(r"incident_id"),             lambda: _rand_id("INC", 7)),
    (re.compile(r"section_id"),              lambda: _rand_id("SEC", 6)),
    (re.compile(r"metric_id"),               lambda: _rand_id("M", 6)),
    (re.compile(r"entity_id"),               lambda: _rand_id("ENT", 8)),
    (re.compile(r"school_id|home_school_id|serving_school_id|prior_year_school_id"),
     lambda: _rand_id("SCH", 8)),
    (re.compile(r"district_id"),             lambda: _rand_id("D", 8)),
    (re.compile(r"admin_ID"),                lambda: _rand_id("ADM", 8)),
    (re.compile(r"teacher_EIN|employee_EIN|principal_EIN"), _rand_ein),
    (re.compile(r"AI_code"),                 lambda: _rand_id("AI", 6)),
    (re.compile(r"IPEDS_identifier"),        lambda: _rand_id("IPEDS", 6)),
    (re.compile(r"course_id"),               lambda: _rand_id("CRS", 6)),

    # ── Codes & classifications ───────────────────────────────────────────
    (re.compile(r"RCDTS_code|school_RCDTS"),  _rand_rcdts),
    (re.compile(r"state_course_id"),          _rand_course_id),
    (re.compile(r"CIP_code|current_CIP_code|retired_CIP_code"), _rand_cip_code),
    (re.compile(r"exit_type_code|exit_code|prior_year_exit_code"), _rand_exit_code),
    (re.compile(r"RNVTA_code"),              lambda: f"{random.randint(1,25):02d}"),
    (re.compile(r"suppression_code"),        lambda: random.choice(["", "01", "02", "03"])),
    (re.compile(r"AccessTestCode"),          lambda: "10"),
    (re.compile(r"exemption_code"),          lambda: f"{random.randint(1,40):02d}"),
    (re.compile(r"fund_code"),               lambda: random.choice(["A","B","C","L","N","P","U"])),
    (re.compile(r"attendance_code"),         lambda: random.choice(["P","A","E","T","H","R"])),
    (re.compile(r"action_code|discipline_action_code"), lambda: f"{random.randint(1,7):02d}"),
    (re.compile(r"incident_type_code"),      lambda: random.choice(["TOB","ALC","DRG","VIO","WPN","OTH"])),
    (re.compile(r"role_code"),               lambda: random.choice(["TCHR","PRIN","APRIN","COUN","LIBR","PARA","SUPP"])),
    (re.compile(r"status_code|enrollment_status_code"), _rand_status_code),
    (re.compile(r"senate_district_code"),    lambda: str(random.randint(1, 59))),
    (re.compile(r"house_district_code"),     lambda: str(random.randint(1, 118))),
    (re.compile(r"region_code"),             lambda: f"{random.randint(1,99):02d}"),
    (re.compile(r"county_code"),             lambda: f"{random.randint(1,102):03d}"),
    (re.compile(r"district_code"),           lambda: f"{random.randint(1,9999):04d}"),
    (re.compile(r"type_code"),               lambda: random.choice(["02","04","08"])),
    (re.compile(r"school_code"),             lambda: f"{random.randint(1,9999):04d}"),
    (re.compile(r"subject_area_code"),       lambda: random.choice(["01","02","03","04","23","51","52","53","54"])),
    (re.compile(r"education_environment_code"), lambda: random.choice(["A","B","C","D","E"])),
    (re.compile(r"group_id|current_group"),  lambda: random.choice([1,2,3,4,5])),
    (re.compile(r"endorsement_code|license_endorsement_code|endorsement_subject_code"), _rand_endorsement_code),
    (re.compile(r"course_level_code|course_level"), lambda: random.choice(["Regular","Honors","AP","IB"])),
    (re.compile(r"accelerated_type_code"),   lambda: f"{random.randint(1,6):02d}"),
    (re.compile(r"gifted_code"),             lambda: f"{random.randint(1,5):02d}"),
    (re.compile(r"disability_category"),     lambda: random.choice(["SLD","SLI","ED","ID","AUT","OHI"])),
    (re.compile(r"cohort_type"),             lambda: random.choice(["4-year","5-year","6-year"])),

    # ── Dates ─────────────────────────────────────────────────────────────
    (re.compile(r"date|Date"),               _rand_date),

    # ── Names / descriptions ──────────────────────────────────────────────
    (re.compile(r"student_name"),            fake.name),
    (re.compile(r"institution_name"),        lambda: f"{fake.city()} {random.choice(['College','University','Community College'])}"),
    (re.compile(r"employer_name"),           fake.company),
    (re.compile(r"course_title|state_course_title"), lambda: random.choice(["Algebra I","English 10","Biology","US History","Art I","Band","Chemistry","Geometry","Pre-Calculus","Spanish I"])),
    (re.compile(r"exam_name"),               lambda: random.choice(["AP Calculus AB","AP English Language","AP US History","AP Biology","AP Chemistry","AP Statistics"])),
    (re.compile(r"program_name"),            lambda: random.choice(["AGRICULTURE AND ENVIRONMENTAL SYSTEMS","HEALTH PROFESSIONS","BUSINESS SYSTEMS","ARTS AND COMMUNICATION","COMPUTER AND INFORMATION SCIENCES"])),
    (re.compile(r"program_description"),     fake.sentence),
    (re.compile(r"cluster_name"),            lambda: random.choice(["Agriculture, Food & Natural Resources","Health Science","Business Management & Administration","Information Technology","Manufacturing"])),
    (re.compile(r"area_name|CCPE_area"),     lambda: random.choice(["METT","Human and Public Services","Finance and Business Services","Health Science and Technology"])),
    (re.compile(r"specific_fields_of_study"),lambda: ", ".join(random.sample(["01.0101","11.0201","51.0913","52.0201","46.0000"], k=3))),
    (re.compile(r"description|status_description|exit_type_description|action_description|incident_description|fund_description|role_description|gifted_description|subject_area_description"), fake.sentence),
    (re.compile(r"scale_name"),              lambda: random.choice(["d1_scaled","d2_scaled","d3_scaled"])),
    (re.compile(r"readiness_measures_subset"), lambda: random.choice(["ATL-REG/SED subset","Original LLD subset","Alternate LLD Subset","Math Subset"])),
    (re.compile(r"endorsement_type"),        lambda: random.choice(["College and Career","STEM","Arts and Humanities"])),
    (re.compile(r"EE_group"),                lambda: random.choice(["Early Childhood","School Age"])),

    # ── Public / private / operational indicators ────────────────────────
    (re.compile(r"public_private_indicator"), lambda: random.choice(["Public", "Public", "Public", "Private"])),
    (re.compile(r"operational_status"),       lambda: random.choice(["Open", "Open", "Open", "Open", "Closed"])),

    # ── Enumerations / choices ────────────────────────────────────────────
    (re.compile(r"school_category"),         lambda: random.choice([2, 4, 8])),
    (re.compile(r"school_type|district_type"), lambda: random.choice(["Elementary","Middle","High School","Unit District"])),
    (re.compile(r"entity_type"),             lambda: random.choice(["school","district","state"])),
    (re.compile(r"assessment_type"),         lambda: random.choice(["IAR","SAT","ISA","DLM-AA"])),
    (re.compile(r"designation_type"),        _rand_designation),
    (re.compile(r"prior_year_status"),       lambda: random.choice(["Comprehensive","Targeted","Additional Targeted","None"])),
    (re.compile(r"Title_I_status_subtype"),  lambda: random.choice(["Targeted","Schoolwide","Eligible but Not Served","Ineligible Due to Ranking"])),
    (re.compile(r"site_type"),               lambda: random.choice(["OP","PK","NP","DP"])),
    (re.compile(r"institution_level"),       _rand_institution_level),
    (re.compile(r"institution_type"),        _rand_institution_type),
    (re.compile(r"snapshot_type"),           lambda: random.choice(["Fall","End of Year"])),
    (re.compile(r"transfer_direction"),      lambda: random.choice(["In","Out"])),
    (re.compile(r"term_type"),               lambda: random.choice(["FY","S1","S2","T1","T2","T3","Q1","Q2","Q3","Q4","SU"])),
    (re.compile(r"attendance_type"),         lambda: random.choice(["In Person","ELearning","Remote","Detention","Excused Absence","Unexcused Absence"])),
    (re.compile(r"SGP_type"),               lambda: random.choice(["cohort","baseline"])),
    (re.compile(r"EL_status"),              lambda: random.choice(["Current EL","Transition Incomplete","Former EL","Never EL"])),
    (re.compile(r"EL_transition_status"),   lambda: random.choice(["Transition Year 1","Transition Year 2","Transition Complete","N/A"])),
    (re.compile(r"FRL_status"),             lambda: random.choice(["Free","Reduced","Not Eligible"])),
    (re.compile(r"grant_type"),             lambda: random.choice(["Title I","Title II","Title III","1003a","1003g"])),
    (re.compile(r"certification_status"),   lambda: random.choice(["Active","Expired","Pending"])),
    (re.compile(r"certification_area"),     lambda: random.choice(["Generalist","ELA","Math","Science","Career and Technical Education"])),
    (re.compile(r"license_type"),           lambda: random.choice(["Professional Educator License","Substitute","Short-Term Approval","Provisional"])),
    (re.compile(r"education_level"),        lambda: random.choice(["Bachelor's","Master's","Doctorate","Specialist"])),
    (re.compile(r"metric_type_classification"), lambda: random.choice(["Accountability","Academic Progress","Environment","Student","Teacher","Financial"])),
    (re.compile(r"career_area_of_interest"),lambda: random.choice(["STEM","Health Sciences","Business","Education","Arts"])),

    # ── Demographics ──────────────────────────────────────────────────────
    (re.compile(r"^race$|race_ethnicity"),  _rand_race),
    (re.compile(r"^gender$"),               _rand_gender),
    (re.compile(r"^state$"),                lambda: "IL"),

    # ── Scores ────────────────────────────────────────────────────────────
    (re.compile(r"scale_score_ELA"),        lambda: _rand_scale_score(200, 800)),
    (re.compile(r"scale_score_Math"),       lambda: _rand_scale_score(200, 800)),
    (re.compile(r"SAT_composite_score"),    lambda: _rand_scale_score(400, 1600)),
    (re.compile(r"SAT_ELA_score"),          lambda: _rand_scale_score(200, 800)),
    (re.compile(r"SAT_Math_score"),         lambda: _rand_scale_score(200, 800)),
    (re.compile(r"ACT_composite_score"),    lambda: random.randint(1, 36)),
    (re.compile(r"ACT_ELA_score"),          lambda: random.randint(1, 36)),
    (re.compile(r"ACT_Math_score"),         lambda: random.randint(1, 36)),
    (re.compile(r"scale_score"),            lambda: _rand_scale_score(100, 900)),
    (re.compile(r"composite_score|ACCESS_composite_score"), lambda: round(random.uniform(1.0, 6.0), 1)),
    (re.compile(r"listening_score|speaking_score|reading_score|writing_score"), lambda: round(random.uniform(1.0, 6.0), 1)),
    (re.compile(r"average_score"),          lambda: _rand_scale_score(100, 500)),
    (re.compile(r"d[123]_scaled_score"),    lambda: random.randint(300, 800)),
    (re.compile(r"ASVAB_score"),            lambda: random.randint(1, 99)),
    (re.compile(r"exam_score"),             lambda: random.randint(1, 5)),
    (re.compile(r"SGP_value"),              lambda: random.randint(1, 99)),
    (re.compile(r"prior_year_scale_score|current_year_scale_score"), lambda: _rand_scale_score(100, 900)),
    (re.compile(r"GPA"),                    lambda: round(random.uniform(0.0, 4.0), 2)),

    # ── Performance levels ────────────────────────────────────────────────
    (re.compile(r"performance_level_ELA|performance_level_Math"), lambda: _rand_performance_level(4)),
    (re.compile(r"performance_level"),       lambda: _rand_performance_level(5)),
    (re.compile(r"evaluation_rating"),       lambda: random.choice(["Excellent","Proficient","Needs Improvement","Unsatisfactory"])),

    # ── Default grade range (sensible low/high) ─────────────────────────
    # low restricted to PK–3 so it's always <= any high (5–12)
    (re.compile(r"default_grade_low"),       lambda: random.choice(["PK", "K", "1", "2", "3"])),
    (re.compile(r"default_grade_high"),      lambda: random.choice(["5", "8", "12"])),

    # ── Grade / year ──────────────────────────────────────────────────────
    (re.compile(r"grade_at_testing|^grade$|grade_level|prior_year_grade"), _rand_grade),
    (re.compile(r"grades_offered"),          lambda: "PK-12"),
    (re.compile(r"school_year|fiscal_year|academic_year|exam_year|score_year|survey_year|data_year|evaluation_year|designation_year|status_year|test_year"), _rand_school_year),
    (re.compile(r"first_9th_grade_year|cohort_year|cohort_year_\d"), lambda: _rand_year()),
    (re.compile(r"^year$"),                  _rand_year),
    (re.compile(r"employment_quarter"),      lambda: random.choice(["Q1","Q2","Q3","Q4"])),

    # ── Grades (letter) ───────────────────────────────────────────────────
    (re.compile(r"course_letter_grade|ELA_course_letter_grade|math_course_letter_grade"), _rand_letter_grade),
    (re.compile(r"grade_description"),       lambda: random.choice(["Student received course term credit","Satisfactory","Unsatisfactory"])),

    # ── Numerator / Denominator / Count ──────────────────────────────────
    (re.compile(r"^numerator$"),             lambda: _rand_count(0, 500)),
    (re.compile(r"^denominator$"),           lambda: _rand_count(10, 1000)),
    (re.compile(r"^count$"),                 lambda: _rand_count(0, 500)),

    # ── Counts ────────────────────────────────────────────────────────────
    (re.compile(r"_count$|_count_"),         lambda: _rand_count(0, 500)),

    # ── Rates / percentages ───────────────────────────────────────────────
    (re.compile(r"_rate$|_rate_|percentage|_percent$|_percent_|achievement_level_percentages"), _rand_percent),
    (re.compile(r"PDA$|^PDA$"),              lambda: round(random.uniform(50, 100), 1)),

    # ── Dollar amounts ────────────────────────────────────────────────────
    (re.compile(r"salary|expenditure|revenue|allocation_amount|grant_amount|EAV|receipts|resources|target$|adequacy_target|local_capacity_target|real_receipts|final_resources"), _rand_dollar),
    (re.compile(r"tax_rate"),                lambda: round(random.uniform(1.0, 8.0), 4)),
    (re.compile(r"credit_hours|course_credit"), lambda: round(random.choice([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]), 2)),

    # ── FTE ───────────────────────────────────────────────────────────────
    (re.compile(r"FTE|enrollment_FTE"),      _rand_fte),

    # ── Days ──────────────────────────────────────────────────────────────
    (re.compile(r"total_school_days"),       lambda: random.randint(170, 185)),
    (re.compile(r"duration_days"),           lambda: random.randint(1, 30)),
    (re.compile(r"_days$"),                  lambda: random.randint(0, 180)),
    (re.compile(r"years_experience|years_as_EL|employment_months|community_service_hours"), lambda: random.randint(0, 30)),

    # ── Indicators / flags (boolean-like) ─────────────────────────────────
    (re.compile(r"_indicator$|_indicator_|_flag$|_flag_"), _rand_flag),

    # ── Tier ──────────────────────────────────────────────────────────────
    (re.compile(r"^tier$"),                  _rand_tier),

    # ── Subject ───────────────────────────────────────────────────────────
    (re.compile(r"^subject$|subject_area"),  _rand_subject),

    # ── Age ───────────────────────────────────────────────────────────────
    (re.compile(r"^age$"),                   lambda: random.randint(3, 21)),

    # ── Nine-month ADA ────────────────────────────────────────────────────
    (re.compile(r"nine_month_ADA"),          lambda: round(random.uniform(100, 50000), 1)),

    # ── Average PE days ───────────────────────────────────────────────────
    (re.compile(r"average_PE"),              lambda: round(random.uniform(0, 5), 2)),

    # ── Parent district ───────────────────────────────────────────────────
    (re.compile(r"parent_district"),         lambda: _rand_id("D", 8)),

    # ── Catch-all for anything unmatched ──────────────────────────────────
    (re.compile(r".*"),                      lambda: fake.pystr(min_chars=4, max_chars=12)),
]


def _generate_value(field_name: str):
    """Pick the first matching generator for *field_name* and call it."""
    for pattern, gen in _FIELD_GENERATORS:
        if pattern.search(field_name):
            return gen()
    return fake.pystr(min_chars=4, max_chars=12)  # fallback


# ── Public API ────────────────────────────────────────────────────────────────

def get_data(
    data_item: str,
    num_rows: int = 1,
    schema: dict | None = None,
) -> list[dict]:
    """Return *num_rows* dicts of mock data for a given data-schema item.

    Parameters
    ----------
    data_item : str
        Exact key **or** case-insensitive substring of a key in
        ``data_schema.json``.
    num_rows : int, optional
        How many records to generate (default 1).
    schema : dict, optional
        Override schema dict (mainly for testing); defaults to the
        module-level ``_SCHEMA``.

    Returns
    -------
    list[dict]
        Each dict has the field names as keys and Faker-generated
        values as values.

    Raises
    ------
    KeyError
        If *data_item* cannot be matched to any key in the schema.
    """
    schema = schema or _SCHEMA

    # ── Resolve key (exact → case-insensitive → substring) ────────────────
    if data_item in schema:
        key = data_item
    else:
        lower = data_item.lower()
        matches = [k for k in schema if lower == k.lower()]
        if not matches:
            matches = [k for k in schema if lower in k.lower()]
        if not matches:
            available = "\n  ".join(sorted(schema.keys()))
            raise KeyError(
                f"No schema entry matches '{data_item}'.\n"
                f"Available keys:\n  {available}"
            )
        if len(matches) > 1:
            print(f"[info] Multiple matches for '{data_item}': {matches}. "
                  f"Using first: '{matches[0]}'")
        key = matches[0]

    fields: list[str] = schema[key]["fields"]
    rows: list[dict] = []
    for _ in range(num_rows):
        row = {field: _generate_value(field) for field in fields}
        rows.append(row)

    return rows


def get_all_data(num_rows: int = 1) -> dict[str, list[dict]]:
    """Generate *num_rows* of mock data for every item in the schema.

    Returns a dict keyed by schema item name.
    """
    return {key: get_data(key, num_rows) for key in _SCHEMA}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args:
        # Demo: one row for every data source
        all_data = get_all_data(num_rows=1)
        print(json.dumps(all_data, indent=2, default=str))
        return

    data_item = args[0]
    num_rows = int(args[1]) if len(args) > 1 else 1

    try:
        rows = get_data(data_item, num_rows)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(rows, indent=2, default=str))


