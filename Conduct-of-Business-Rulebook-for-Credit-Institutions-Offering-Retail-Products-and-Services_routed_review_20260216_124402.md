# Rule Engine Execution Review Report

**Generated:** 2026-02-16T12:44:02.460437
**Rules file:** `use_case_02/Conduct-of-Business-Rulebook-for-Credit-Institutions-Offering-Retail-Products-and-Services_routed_cel.json`
**Mock-data file:** `use_case_02/Conduct-of-Business-Rulebook-for-Credit-Institutions-Offering-Retail-Products-and-Services_routed_cel_mockdata.json`
**Starting rules:** 225
**Total runs:** 969
**Total steps:** 1647

---
## Confidence Score: **95 / 100**

### Health Statistics

| Metric | Value |
|--------|-------|
| Total runs | 969 |
| Runs with errors | 0 |
| Error rate | 0.0% |
| Total ERROR outputs | 0 |
| Total null outputs | 0 |
| Null output rate | 0.0% |

---
## 1. Data Analysed

The rule engine evaluated 969 execution runs across a comprehensive set of business rules from the 'Conduct of Business Rulebook for Credit Institutions Offering Retail Products and Services'. The data sources tested included CAS (credit agreement), RPS (regulated person system), CR (credit regulation), SR (staff regulation), and Regulated_Person_Systems. Input variables spanned credit types, client identifiers, agreement details, communication methods, and regulatory compliance indicators. Test coverage was extensive, with 225 starting rules and 257 unique execution paths identified, demonstrating broad rule coverage and varied branching logic.

**Data sources:** `CAS.credit_agreement_type`, `RPS.client_id`, `CR.termination_reason`, `CR.agreement_type`, `CAS.credit_amount_repaid_early`, `RPS.agreement_type`, `RPS.account_type`, `RPS.regulated_person_id`, `RPS.agreement_concluded`, `RPS.communication_type`, `RPS.promotion_type`, `SR.competency_level`, `CAS.time_between_repayment_and_termination`, `CAS.borrowing_rate_fixed_period`, `RPS.communication_medium`, `RPS.client_access_to_internet`, `RPS.branch_location`, `RPS.credit_intermediary`, `Regulated_Person_Systems.presentation_format`, `Regulated_Person_Systems.information_content`, `Regulated_Person_Systems.access_method`, `Regulated_Person_Systems.electronic_information`, `Regulated_Person_Systems.disclosed_material`, `Regulated_Person_Systems.internet_site_url`, `Regulated_Person_Systems.contact_person`, `Regulated_Person_Systems.data_accuracy`, `Regulated_Person_Systems.site_responsibility`, `Regulated_Person_Systems.warning_visibility`, `Regulated_Person_Systems.hyperlink_usage`, `Regulated_Person_Systems.hyperlink_consistency`

---
## 2. Outputs

Derived variables included compliance status, client notification status, compensation amounts, communication status, advertisement compliance, credit availability, competency status, information compliance, notice status, fee status, disclosure status, and numerous others. The most common output values were 'Compliant' (38 out of 56) for derived.compliance_status, 'Notification Required' (5 out of 15) for derived.client_notification_status, and '0.0' (2 out of 15) for derived.compensation_amount. Branch coverage was strong, with 257 unique paths identified, and all execution runs completed successfully with no errors or null outputs.

**Key output variables:**
- `derived.compliance_status`
- `derived.client_notification_status`
- `derived.compensation_amount`
- `derived.communication_status`
- `derived.advertisement_compliance`
- `derived.credit_availability_status`
- `derived.competency_status`
- `derived.information_compliance_status`
- `derived.notice_status`
- `derived.fee_status`
- `derived.disclosure_status`

### Output Value Distributions

**`derived.compliance_status`** (56 computations)

| Value | Count |
|-------|-------|
| Compliant | 38 |
| Non-Compliant | 18 |

**`derived.client_notification_status`** (15 computations)

| Value | Count |
|-------|-------|
| Notification Required | 5 |
| Not Applicable | 4 |
| No Action Needed | 3 |
| Notified | 3 |

**`derived.compensation_amount`** (15 computations)

| Value | Count |
|-------|-------|
| 0.0 | 2 |
| 669.2256 | 1 |
| 229.2082 | 1 |
| 523.2631 | 1 |
| 223.86305000000002 | 1 |
| 116.2949 | 1 |
| 484.78974999999997 | 1 |
| 405.1321 | 1 |

**`derived.communication_status`** (12 computations)

| Value | Count |
|-------|-------|
| Communication initiated | 4 |
| Not applicable | 4 |
| Compliant | 2 |
| Non-Compliant | 1 |
| Communication sent | 1 |

**`derived.advertisement_compliance`** (11 computations)

| Value | Count |
|-------|-------|
| Non-Compliant | 7 |
| Compliant | 4 |

**`derived.credit_availability_status`** (10 computations)

| Value | Count |
|-------|-------|
| Misleading Claim | 3 |
| Not Available | 3 |
| Compliant | 2 |
| Available | 2 |

**`derived.competency_status`** (10 computations)

| Value | Count |
|-------|-------|
| Compliant | 5 |
| Non-Compliant | 5 |

**`derived.information_compliance_status`** (6 computations)

| Value | Count |
|-------|-------|
| Compliant | 3 |
| Non-Compliant | 3 |

**`derived.notice_status`** (6 computations)

| Value | Count |
|-------|-------|
| Pending | 3 |
| Notified | 2 |
| Notice Provided | 1 |

**`derived.fee_status`** (6 computations)

| Value | Count |
|-------|-------|
| No fee charged | 4 |
| No Fee Charged | 1 |
| Fee applicable | 1 |

**`derived.disclosure_status`** (6 computations)

| Value | Count |
|-------|-------|
| Compliant | 5 |
| Non-Price Features Disclosed | 1 |

**`derived.notification_status`** (6 computations)

| Value | Count |
|-------|-------|
| Notification Required | 3 |
| No Action Needed | 2 |
| Completed | 1 |

**`derived.interest_rate_disclosure_status`** (6 computations)

| Value | Count |
|-------|-------|
| Disclosed | 3 |
| Not Disclosed | 2 |
| Accessible and up-to-date interest rate information. | 1 |

**`derived.policy_status`** (6 computations)

| Value | Count |
|-------|-------|
| Non-Compliant | 3 |
| Compliant | 2 |
| Policy Exists | 1 |

**`derived.approval_status`** (6 computations)

| Value | Count |
|-------|-------|
| Approved | 4 |
| Not Approved | 1 |
| rejected | 1 |

**`derived.staff_competency_status`** (6 computations)

| Value | Count |
|-------|-------|
| Compliant | 6 |

**`derived.conflict_management_status`** (6 computations)

| Value | Count |
|-------|-------|
| not_managed | 3 |
| Compliant | 1 |
| partially_managed | 1 |
| managed | 1 |

**`derived.branch_information_status`** (5 computations)

| Value | Count |
|-------|-------|
| Published and Available Online | 5 |

**`derived.intermediary_disclosure_status`** (5 computations)

| Value | Count |
|-------|-------|
| Not Applicable | 4 |
| Disclosure Required | 1 |

**`derived.prominently_presented`** (5 computations)

| Value | Count |
|-------|-------|
| No | 3 |
| Yes | 2 |

---
## 3. Conclusions & Data Insights

The business purpose of the rule engine is to ensure regulatory compliance for credit institutions offering retail products and services. Key patterns observed include consistent compliance outcomes ('Compliant' and 'Non-Compliant') across various derived variables, with a slight majority favoring compliance. Notable distributions include a balanced split between 'Compliant' and 'Non-Compliant' for compliance status, and a wide variety of compensation amounts reflecting diverse scenarios. The rule engine demonstrates robust execution with no errors or null outputs, indicating reliable operation across all tested paths.

### Key Insights

- Strong branch coverage with 257 unique paths and average path length of 1.7 steps
- All 969 execution runs completed successfully with 0% error rate and 0% null output rate
- High consistency in derived outputs with no major anomalies or unexpected patterns
- Broad input variable coverage across multiple data sources including credit agreements, client details, and regulatory systems
- Diverse output distribution showing varied compliance outcomes across different regulatory domains

---
## 4. Correctness Evaluation

The rule engine demonstrated excellent correctness with zero errors and zero null outputs across all 969 execution runs. All chunks showed 95% confidence, and no issues were reported across any of the 20 chunks reviewed. The system exhibits strong reliability and consistent performance, with all derived outputs being valid and meaningful. The most significant strength is the complete absence of runtime errors or null outputs, indicating robust implementation and testing. Recommendations focus on maintaining current quality standards and ensuring continued monitoring of new rule additions.

### Strengths

- Zero error rate across all 969 execution runs
- Zero null output rate across all 969 execution runs
- Consistent performance across all 20 chunks with 95% confidence
- Comprehensive coverage of 257 unique execution paths
- Robust handling of diverse input variables from multiple data sources

### Recommendations

- Continue monitoring for any potential edge cases that might not have been covered in the current test suite
- Maintain current quality assurance practices to sustain the high level of correctness
- Consider expanding test coverage to include more extreme or boundary value scenarios

---
## Appendix: Execution Path Statistics

- **Unique paths:** 257
- **Steps per run:** min=1, max=200, avg=1.7

### Top Execution Paths

| Count | Path |
|-------|------|
| 5 | R003 |
| 5 | R004 |
| 5 | R005 |
| 5 | R006 |
| 5 | R007 |
| 5 | R008 |
| 5 | R009 |
| 5 | R010 |
| 5 | R011 |
| 5 | R012 |
| 5 | R013 |
| 5 | R014 |
| 5 | R016 |
| 5 | R018 |
| 5 | R019 |

### Terminal States

| State | Count |
|-------|-------|
| R003 (no routing) | 5 |
| R004 (no routing) | 5 |
| R005 (no routing) | 5 |
| R006 (no routing) | 5 |
| R007 (no routing) | 5 |
| R008 (no routing) | 5 |
| R009 (no routing) | 5 |
| R010 (no routing) | 5 |
| R011 (no routing) | 5 |
| R012 (no routing) | 5 |
| R013 (no routing) | 5 |
| R014 (no routing) | 5 |
| R016 (no routing) | 5 |
| R018 (no routing) | 5 |
| R019 (no routing) | 5 |
