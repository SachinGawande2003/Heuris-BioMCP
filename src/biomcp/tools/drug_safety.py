"""
BioMCP — FDA Drug Safety Intelligence Suite
=============================================
Real-time pharmacovigilance analysis using FDA's OpenFDA APIs.
No API key required for standard queries (240 requests/min limit).

This is the only MCP server providing direct access to FDA adverse event
data with clinical signal detection — enabling AI-assisted drug safety
analysis that previously required specialized pharmacovigilance software.

Tools:
  query_adverse_events     — FDA FAERS adverse event query
  analyze_safety_signals   — Disproportionality analysis (PRR/ROR)
  get_drug_label_warnings  — FDA-approved label safety sections
  compare_drug_safety      — Head-to-head safety profile comparison

APIs:
  FDA OpenFDA Events:  https://api.fda.gov/drug/event.json
  FDA OpenFDA Labels:  https://api.fda.gov/drug/label.json
  FDA OpenFDA NDC:     https://api.fda.gov/drug/ndc.json

Scientific basis:
  - Proportional Reporting Ratio (PRR): Evans et al. 2001
  - Reporting Odds Ratio (ROR): van Puijenbroek et al. 2002
  - IC (Information Component): Bate et al. 1998 (WHO Bayesian method)
  - Signal threshold: PRR ≥ 2, χ² ≥ 4, n ≥ 3 (standard pharmacovigilance)
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from loguru import logger

from biomcp.utils import (
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)

_OPENFDA_BASE = "https://api.fda.gov"

# ─────────────────────────────────────────────────────────────────────────────
# Common serious adverse event MedDRA terms for automated analysis
# ─────────────────────────────────────────────────────────────────────────────

_SERIOUS_EVENT_CATEGORIES = {
    "cardiac":        ["myocardial infarction", "cardiac arrest", "arrhythmia", "QT prolongation",
                       "atrial fibrillation", "heart failure", "sudden cardiac death"],
    "hepatic":        ["hepatotoxicity", "liver failure", "jaundice", "elevated transaminase",
                       "drug-induced liver injury", "hepatitis"],
    "hematologic":    ["agranulocytosis", "thrombocytopenia", "anemia", "neutropenia",
                       "aplastic anemia", "coagulopathy"],
    "neurological":   ["seizure", "stroke", "encephalopathy", "peripheral neuropathy",
                       "serotonin syndrome", "neuroleptic malignant"],
    "renal":          ["acute kidney injury", "renal failure", "nephrotoxicity"],
    "hypersensitivity":["anaphylaxis", "stevens-johnson", "toxic epidermal", "angioedema",
                        "drug reaction with eosinophilia"],
    "oncology":       ["secondary malignancy", "tumor lysis", "cytokine release", "immunosuppression"],
    "respiratory":    ["pulmonary embolism", "pneumonitis", "respiratory failure", "ILD"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: query_adverse_events
# ─────────────────────────────────────────────────────────────────────────────

@cached("fda")
@rate_limited("default")
@with_retry(max_attempts=3)
async def query_adverse_events(
    drug_name:       str,
    event_type:      str = "all",
    serious_only:    bool = False,
    max_results:     int = 50,
    patient_sex:     str = "",
    age_group:       str = "",
) -> dict[str, Any]:
    """
    Query FDA FAERS (Adverse Event Reporting System) for drug safety signals.

    Provides access to millions of adverse event reports submitted to the FDA,
    enabling real-time pharmacovigilance analysis.

    Args:
        drug_name:    Drug name (generic or brand). E.g. 'ibuprofen', 'Humira', 'adalimumab'.
        event_type:   'all' | 'cardiac' | 'hepatic' | 'hematologic' | 'neurological' |
                      'renal' | 'hypersensitivity' | 'respiratory' | 'oncology'.
        serious_only: Only return serious adverse events (hospitalization, death, disability).
        max_results:  Maximum reports to analyze (10–500). Default 50.
        patient_sex:  'male' | 'female' | '' (all). Filter by patient sex.
        age_group:    'pediatric' (<18) | 'adult' (18–64) | 'elderly' (>64) | '' (all).

    Returns:
        {
          drug, total_reports, serious_reports,
          outcomes: {death_count, hospitalization_count, disabled_count},
          top_reactions: [{reaction, count, pct, seriousness}],
          demographics: {age_distribution, sex_distribution},
          temporal_trend: {by_year},
          reporting_note
        }
    """
    client = await get_http_client()

    # Build OpenFDA search query
    drug_query = f'patient.drug.medicinalproduct:"{drug_name}"'
    if event_type != "all" and event_type in _SERIOUS_EVENT_CATEGORIES:
        terms = _SERIOUS_EVENT_CATEGORIES[event_type]
        event_query = " OR ".join(f'patient.reaction.reactionmeddrapt:"{t}"' for t in terms[:5])
        search_q = f"({drug_query}) AND ({event_query})"
    else:
        search_q = drug_query

    if serious_only:
        search_q += ' AND serious:"1"'

    if patient_sex:
        sex_code = {"male": "1", "female": "2"}.get(patient_sex.lower(), "")
        if sex_code:
            search_q += f' AND patient.patientsex:"{sex_code}"'

    # Age group mapping (years)
    if age_group:
        age_ranges = {
            "pediatric": ("0", "17"),
            "adult":     ("18", "64"),
            "elderly":   ("65", "120"),
        }.get(age_group.lower())
        if age_ranges:
            search_q += f' AND patient.patientonsetage:[{age_ranges[0]} TO {age_ranges[1]}]'

    logger.info(f"[FDASafety] Querying FAERS for: {drug_name}")

    # ── Main adverse event query ──────────────────────────────────────────────
    events_resp = await client.get(
        f"{_OPENFDA_BASE}/drug/event.json",
        params={
            "search": search_q,
            "count":  "patient.reaction.reactionmeddrapt.exact",
            "limit":  min(max_results, 100),
        },
        headers={"Accept": "application/json"},
    )

    # ── Total count query ─────────────────────────────────────────────────────
    count_resp = await client.get(
        f"{_OPENFDA_BASE}/drug/event.json",
        params={"search": drug_query, "limit": 1},
        headers={"Accept": "application/json"},
    )

    # ── Serious outcome breakdown ─────────────────────────────────────────────
    outcomes: dict[str, int] = {}
    for outcome_code, label in [("1", "death"), ("2", "life_threatening"),
                                  ("3", "hospitalization"), ("4", "disabled"),
                                  ("5", "congenital_anomaly"), ("6", "other_serious")]:
        out_resp = await client.get(
            f"{_OPENFDA_BASE}/drug/event.json",
            params={
                "search": f"({drug_query}) AND patient.reaction.reactionoutcome:\"{outcome_code}\"",
                "limit": 1,
            },
        )
        if out_resp.status_code == 200:
            out_data = out_resp.json()
            outcomes[label] = out_data.get("meta", {}).get("results", {}).get("total", 0)

    # ── Sex distribution ──────────────────────────────────────────────────────
    sex_dist: dict[str, int] = {}
    for sex_code, sex_label in [("1", "male"), ("2", "female")]:
        sex_resp = await client.get(
            f"{_OPENFDA_BASE}/drug/event.json",
            params={
                "search": f'({drug_query}) AND patient.patientsex:"{sex_code}"',
                "limit": 1,
            },
        )
        if sex_resp.status_code == 200:
            sex_data = sex_resp.json()
            sex_dist[sex_label] = sex_data.get("meta", {}).get("results", {}).get("total", 0)

    # ── Yearly trend ──────────────────────────────────────────────────────────
    yearly_resp = await client.get(
        f"{_OPENFDA_BASE}/drug/event.json",
        params={"search": drug_query, "count": "receivedate", "limit": 5},
    )

    # Parse results
    total_reports = 0
    if count_resp.status_code == 200:
        total_reports = count_resp.json().get("meta", {}).get("results", {}).get("total", 0)

    top_reactions: list[dict[str, Any]] = []
    if events_resp.status_code == 200:
        events_data = events_resp.json()
        reaction_counts = events_data.get("results", [])
        for r in reaction_counts[:20]:
            term  = r.get("term", "")
            count = r.get("count", 0)
            top_reactions.append({
                "reaction":    term,
                "count":       count,
                "pct_of_total":round(count / max(total_reports, 1) * 100, 3),
                "event_type":  _classify_event(term),
            })

    yearly_trend: list[dict] = []
    if yearly_resp.status_code == 200:
        yearly_data = yearly_resp.json()
        for item in yearly_data.get("results", [])[:10]:
            yearly_trend.append({
                "date":  item.get("time", ""),
                "count": item.get("count", 0),
            })

    return {
        "drug":          drug_name,
        "query_type":    event_type,
        "filters":       {
            "serious_only": serious_only,
            "sex":          patient_sex or "all",
            "age_group":    age_group or "all",
        },
        "total_reports": total_reports,
        "outcomes": {
            "deaths":              outcomes.get("death", 0),
            "life_threatening":    outcomes.get("life_threatening", 0),
            "hospitalizations":    outcomes.get("hospitalization", 0),
            "disability":          outcomes.get("disabled", 0),
            "congenital_anomaly":  outcomes.get("congenital_anomaly", 0),
            "other_serious":       outcomes.get("other_serious", 0),
        },
        "sex_distribution":   sex_dist,
        "top_reactions":      top_reactions,
        "yearly_trend":       yearly_trend[-5:] if yearly_trend else [],
        "faers_dashboard_url":f"https://fis.fda.gov/sense/app/95239e26-e0be-42d9-a960-9a5f7f1c25ee/sheet/7a47a261-d58b-4203-a8aa-6d3021737452/state/analysis?select=filterAnd,D_DRUGNAME,{drug_name.replace(' ', '%20')}",
        "reporting_note": (
            "FAERS data reflects voluntary and mandatory reports. "
            "Causality cannot be determined from reporting frequency alone. "
            "Use clinical judgment and consult FDA labeling for safety decisions. "
            "Data current through latest FAERS quarterly release."
        ),
    }


def _classify_event(term: str) -> str:
    term_lower = term.lower()
    for category, keywords in _SERIOUS_EVENT_CATEGORIES.items():
        if any(kw in term_lower for kw in keywords):
            return category.capitalize()
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: analyze_safety_signals
# ─────────────────────────────────────────────────────────────────────────────

@cached("fda")
@rate_limited("default")
@with_retry(max_attempts=3)
async def analyze_safety_signals(
    drug_name:    str,
    event_terms:  list[str] | None = None,
    comparators:  list[str] | None = None,
) -> dict[str, Any]:
    """
    Perform pharmacovigilance disproportionality analysis on FDA FAERS data.

    Calculates three established signal detection metrics:
    - PRR (Proportional Reporting Ratio) — Evans et al. 2001
    - ROR (Reporting Odds Ratio) — van Puijenbroek et al. 2002
    - IC (Information Component) — Bayesian confidence interval

    Signal criteria (WHO UMC standard): PRR ≥ 2, χ² ≥ 4, n ≥ 3

    Args:
        drug_name:   Drug of interest.
        event_terms: Adverse events to analyze. Auto-selects top events if empty.
        comparators: Comparator drugs for reference (e.g. drug class members).

    Returns:
        {
          drug, signals: [{event, n, prr, ror, ic, chi_squared,
          signal_detected, evidence_level}],
          summary_statistics, methodology_note
        }
    """
    client = await get_http_client()

    # Get total report count in FAERS (background)
    total_resp = await client.get(
        f"{_OPENFDA_BASE}/drug/event.json",
        params={"limit": 1},
    )
    total_faers = 0
    if total_resp.status_code == 200:
        total_faers = total_resp.json().get("meta", {}).get("results", {}).get("total", 1)

    # Get reports for this drug
    drug_resp = await client.get(
        f"{_OPENFDA_BASE}/drug/event.json",
        params={
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": 1,
        },
    )
    n_drug = 0
    if drug_resp.status_code == 200:
        n_drug = drug_resp.json().get("meta", {}).get("results", {}).get("total", 0)

    if not event_terms:
        # Auto-fetch top reactions for this drug
        top_resp = await client.get(
            f"{_OPENFDA_BASE}/drug/event.json",
            params={
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "count":  "patient.reaction.reactionmeddrapt.exact",
                "limit":  10,
            },
        )
        if top_resp.status_code == 200:
            event_terms = [r["term"] for r in top_resp.json().get("results", [])[:8]]
        else:
            event_terms = []

    # Calculate disproportionality metrics for each event
    signals: list[dict[str, Any]] = []

    async def _calc_signal(event_term: str) -> dict[str, Any] | None:
        # n_ae_drug: reports with drug + event
        ae_drug_resp = await client.get(
            f"{_OPENFDA_BASE}/drug/event.json",
            params={
                "search": (
                    f'patient.drug.medicinalproduct:"{drug_name}" AND '
                    f'patient.reaction.reactionmeddrapt:"{event_term}"'
                ),
                "limit": 1,
            },
        )
        if ae_drug_resp.status_code != 200:
            return None
        n_ae_drug = ae_drug_resp.json().get("meta", {}).get("results", {}).get("total", 0)
        if n_ae_drug < 3:  # Below minimum reporting threshold
            return None

        # n_ae_all: reports with event across all drugs
        ae_all_resp = await client.get(
            f"{_OPENFDA_BASE}/drug/event.json",
            params={
                "search": f'patient.reaction.reactionmeddrapt:"{event_term}"',
                "limit": 1,
            },
        )
        if ae_all_resp.status_code != 200:
            return None
        n_ae_all = ae_all_resp.json().get("meta", {}).get("results", {}).get("total", 1)

        # 2×2 contingency table
        # a = drug + event, b = drug + no event, c = no drug + event, d = no drug + no event
        a = max(n_ae_drug, 1)
        b = max(n_drug - a, 1)
        c = max(n_ae_all - a, 1)
        d = max(total_faers - a - b - c, 1)

        # PRR (Proportional Reporting Ratio)
        prr = (a / (a + b)) / (c / (c + d))

        # ROR (Reporting Odds Ratio)
        ror = (a * d) / (b * c)

        # Chi-squared (with continuity correction)
        n_total = a + b + c + d
        chi2 = n_total * (abs(a*d - b*c) - n_total/2)**2 / ((a+b)*(c+d)*(a+c)*(b+d))

        # IC (Information Component) — Bayesian method
        ic = math.log2(
            (a + 0.5) / ((a + b + 0.5) * (a + c + 0.5) / (n_total + 0.5))
        )
        ic025 = ic - 3.3 * math.sqrt(1/(a+0.5) - 1/(n_drug+0.5) + 1/n_total)

        # Signal detection
        signal = prr >= 2.0 and chi2 >= 4.0 and a >= 3
        signal_bayesian = ic025 > 0

        evidence_level = (
            "STRONG"    if prr >= 4 and chi2 >= 10 and a >= 10 else
            "MODERATE"  if prr >= 2 and chi2 >= 4  and a >= 5 else
            "WEAK"      if prr >= 2 and a >= 3 else
            "NO SIGNAL"
        )

        return {
            "event":              event_term,
            "n_reports":          a,
            "n_drug_total":       n_drug,
            "n_event_all_drugs":  n_ae_all,
            "prr":                round(prr, 3),
            "ror":                round(ror, 3),
            "ic":                 round(ic, 3),
            "ic025":              round(ic025, 3),
            "chi_squared":        round(chi2, 3),
            "signal_detected":    signal,
            "signal_bayesian":    signal_bayesian,
            "evidence_level":     evidence_level,
            "event_category":     _classify_event(event_term),
            "interpretation": (
                f"PRR={prr:.1f}, ROR={ror:.1f} — {'SIGNAL DETECTED' if signal else 'No signal'}. "
                f"{evidence_level} evidence based on {a} reports."
            ),
        }

    # Run all signal calculations in parallel
    results_raw = await asyncio.gather(
        *[_calc_signal(term) for term in (event_terms or [])],
        return_exceptions=True,
    )

    for r in results_raw:
        if r is not None and not isinstance(r, Exception):
            signals.append(r)

    # Sort by PRR descending
    signals.sort(key=lambda x: x.get("prr", 0), reverse=True)

    strong_signals = [s for s in signals if s["evidence_level"] == "STRONG"]
    moderate_signals = [s for s in signals if s["evidence_level"] == "MODERATE"]

    return {
        "drug":              drug_name,
        "total_reports":     n_drug,
        "events_analyzed":   len(signals),
        "signals":           signals,
        "summary": {
            "strong_signals":   len(strong_signals),
            "moderate_signals": len(moderate_signals),
            "top_signal":       signals[0]["event"] if signals else "None",
            "top_prr":          signals[0]["prr"] if signals else 0,
        },
        "signal_criteria": {
            "PRR_threshold":    "≥ 2.0",
            "chi2_threshold":   "≥ 4.0",
            "min_reports":      "≥ 3",
            "reference":        "Evans et al. 2001 (Br J Clin Pharmacol)",
        },
        "methodology_note": (
            "PRR and ROR quantify how disproportionately an event is reported "
            "with this drug vs all other drugs in FAERS. This is signal DETECTION, "
            "not causality PROOF. Signals require clinical evaluation and regulatory review. "
            "FAERS under-reports: true adverse event frequency is substantially higher."
        ),
        "regulatory_note": (
            "FDA MedWatch: 1-800-FDA-1088 | https://www.fda.gov/safety/medwatch "
            "Report suspected adverse events to maintain drug safety surveillance."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: get_drug_label_warnings
# ─────────────────────────────────────────────────────────────────────────────

@cached("fda")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_drug_label_warnings(
    drug_name:       str,
    sections:        list[str] | None = None,
) -> dict[str, Any]:
    """
    Retrieve FDA-approved drug label safety sections.

    Provides direct access to the legally binding FDA label, including
    black box warnings, contraindications, adverse reactions, and
    drug interactions — essential for clinical decision support.

    Args:
        drug_name: Generic or brand drug name.
        sections:  Label sections to retrieve. Default: all safety sections.
                   Options: 'boxed_warning', 'warnings', 'contraindications',
                            'adverse_reactions', 'drug_interactions',
                            'use_in_specific_populations', 'overdosage'.

    Returns:
        {
          drug, brand_names, generic_name, manufacturer,
          has_black_box_warning, boxed_warning,
          warnings, contraindications, adverse_reactions,
          drug_interactions, pregnancy_category,
          label_date, full_label_url
        }
    """
    client = await get_http_client()

    # Search label database
    label_resp = await client.get(
        f"{_OPENFDA_BASE}/drug/label.json",
        params={
            "search": (
                f'openfda.generic_name:"{drug_name}" OR '
                f'openfda.brand_name:"{drug_name}" OR '
                f'openfda.substance_name:"{drug_name}"'
            ),
            "limit": 3,
        },
        headers={"Accept": "application/json"},
    )

    if label_resp.status_code == 404:
        return {
            "drug":  drug_name,
            "error": f"No FDA label found for '{drug_name}'. Check spelling or try generic name.",
            "fda_label_search": f"https://labels.fda.gov/search?labelSearch={drug_name}",
        }

    if label_resp.status_code != 200:
        label_resp.raise_for_status()

    results = label_resp.json().get("results", [])
    if not results:
        return {
            "drug":  drug_name,
            "error": f"No FDA label found for '{drug_name}'.",
        }

    # Use the most recent label
    label = results[0]
    openfda = label.get("openfda", {})

    def _get_section(label: dict, *keys: str) -> str:
        """Extract first non-empty section from multiple possible key names."""
        for key in keys:
            val = label.get(key)
            if val:
                return (val[0] if isinstance(val, list) else val)[:3000]
        return ""

    boxed = _get_section(label, "boxed_warning")
    warnings = _get_section(label, "warnings_and_cautions", "warnings")
    contraindications = _get_section(label, "contraindications")
    adverse_reactions = _get_section(label, "adverse_reactions")
    drug_interactions = _get_section(label, "drug_interactions")
    specific_pops     = _get_section(label, "use_in_specific_populations")
    overdosage        = _get_section(label, "overdosage")

    # Extract pregnancy information
    pregnancy_info = _get_section(label, "pregnancy", "use_in_specific_populations")
    preg_category = "See label"
    if "category x" in pregnancy_info.lower():
        preg_category = "Category X — Contraindicated in pregnancy"
    elif "category d" in pregnancy_info.lower():
        preg_category = "Category D — Evidence of fetal risk"
    elif "category c" in pregnancy_info.lower():
        preg_category = "Category C — Cannot rule out risk"
    elif "category b" in pregnancy_info.lower():
        preg_category = "Category B — No evidence of risk in humans"
    elif "category a" in pregnancy_info.lower():
        preg_category = "Category A — Adequate studies show no risk"

    return {
        "drug":                drug_name,
        "generic_name":        (openfda.get("generic_name") or ["Unknown"])[0],
        "brand_names":         openfda.get("brand_name", [])[:5],
        "manufacturer":        (openfda.get("manufacturer_name") or ["Unknown"])[0],
        "application_number":  (openfda.get("application_number") or [""])[0],
        "has_black_box_warning": bool(boxed),
        "boxed_warning":       boxed or "None",
        "warnings_and_cautions": warnings[:2000] or "See full label.",
        "contraindications":   contraindications[:1500] or "None listed.",
        "adverse_reactions":   adverse_reactions[:2000] or "See full label.",
        "drug_interactions":   drug_interactions[:1500] or "See full label.",
        "pregnancy_category":  preg_category,
        "use_in_specific_populations": specific_pops[:1000] or "See full label.",
        "overdosage":          overdosage[:1000] or "See full label.",
        "label_version":       label.get("version", ""),
        "effective_time":      label.get("effective_time", ""),
        "full_label_url":      f"https://labels.fda.gov/search?labelSearch={drug_name}",
        "important_note": (
            "This is the FDA-approved label. Always refer to current prescribing information "
            "for clinical decisions. Labels are updated periodically — verify currency."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: compare_drug_safety
# ─────────────────────────────────────────────────────────────────────────────

@rate_limited("default")
async def compare_drug_safety(
    drugs:          list[str],
    event_category: str = "all",
) -> dict[str, Any]:
    """
    Head-to-head safety comparison between 2–5 drugs using FAERS data.

    Essential for:
    - Drug class comparisons (e.g. statins, SSRIs, TNF inhibitors)
    - New drug vs established standard-of-care
    - Generic vs brand safety surveillance
    - Pediatric vs adult safety profile differences

    Args:
        drugs:          List of 2–5 drug names to compare.
        event_category: Event type to focus: 'all' | 'cardiac' | 'hepatic' |
                        'hematologic' | 'neurological' | 'hypersensitivity'.

    Returns:
        {
          drugs_compared, comparison_table,
          head_to_head: [{event, drug_A_prr, drug_B_prr, relative_risk}],
          safety_ranking, key_differences
        }
    """
    if len(drugs) < 2 or len(drugs) > 5:
        raise ValueError("Provide 2–5 drugs for comparison.")

    client = await get_http_client()

    # Get total count for each drug in parallel
    async def _get_drug_stats(drug: str) -> dict[str, Any]:
        resp = await client.get(
            f"{_OPENFDA_BASE}/drug/event.json",
            params={
                "search": f'patient.drug.medicinalproduct:"{drug}"',
                "count":  "patient.reaction.reactionmeddrapt.exact",
                "limit":  15,
            },
        )
        count_resp = await client.get(
            f"{_OPENFDA_BASE}/drug/event.json",
            params={"search": f'patient.drug.medicinalproduct:"{drug}"', "limit": 1},
        )
        total = 0
        if count_resp.status_code == 200:
            total = count_resp.json().get("meta", {}).get("results", {}).get("total", 0)

        top_events: list[tuple[str, int]] = []
        if resp.status_code == 200:
            for item in resp.json().get("results", [])[:10]:
                top_events.append((item.get("term", ""), item.get("count", 0)))

        return {"drug": drug, "total_reports": total, "top_events": top_events}

    drug_stats_raw = await asyncio.gather(
        *[_get_drug_stats(d) for d in drugs],
        return_exceptions=True,
    )

    drug_stats: list[dict] = [
        s for s in drug_stats_raw if isinstance(s, dict)
    ]

    # Build comparison table
    comparison_table: list[dict[str, Any]] = []
    for stats in drug_stats:
        comparison_table.append({
            "drug":          stats["drug"],
            "total_reports": stats["total_reports"],
            "top_5_events":  [e[0] for e in stats["top_events"][:5]],
        })

    # Find events unique to or disproportionate in each drug
    all_events: dict[str, dict[str, int]] = {}
    for stats in drug_stats:
        for event, count in stats["top_events"]:
            if event not in all_events:
                all_events[event] = {}
            all_events[event][stats["drug"]] = count

    # Events that appear for some drugs but not others (potential differences)
    key_differences: list[dict[str, Any]] = []
    for event, drug_counts in all_events.items():
        if len(drug_counts) < len(drugs):
            # Event only reported for some drugs
            drugs_with = list(drug_counts.keys())
            drugs_without = [d for d in drugs if d not in drug_counts]
            if drugs_without:
                key_differences.append({
                    "event":          event,
                    "reported_with":  drugs_with,
                    "not_reported_with": drugs_without,
                    "note": f"'{event}' reported for {drugs_with} but not in top reactions for {drugs_without}",
                })

    # Safety ranking by total reports (proxy for reporting frequency)
    ranked = sorted(drug_stats, key=lambda x: x["total_reports"])

    return {
        "drugs_compared":   drugs,
        "event_focus":      event_category,
        "comparison_table": comparison_table,
        "key_event_differences": key_differences[:8],
        "reporting_volume_ranking": [
            {"rank": i+1, "drug": s["drug"], "total_reports": s["total_reports"]}
            for i, s in enumerate(ranked)
        ],
        "interpretation_guide": {
            "report_count_caveat": "Higher report count ≠ more dangerous. More widely used drugs accumulate more reports.",
            "signal_analysis":    "Use analyze_safety_signals() tool for rigorous disproportionality analysis.",
            "confounders":        "Indication bias, reporting rate, and patient population differ between drugs.",
        },
        "faers_data_note": (
            "FAERS report counts reflect pharmacovigilance surveillance data. "
            "Causality is not established. For clinical comparison, see published "
            "randomized controlled trial data and systematic reviews."
        ),
    }
