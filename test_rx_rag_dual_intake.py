"""Regression tests for rx_rag_dual_intake.py text extraction (run: python3 -m pytest -q)."""
import re

import pytest

import rx_rag_dual_intake as rx

SAMPLE_RX = """Name: Ramesh Kumar Age: 45 Sex: M
Date: 12/10/2026
Address: Pune
BP: 130/80  Temp: 101.2 F
40 Gark & Smafly x 5 days
Cough since 3 days
h/o Chest X-ray
Sputum for (AFB)
Rx
T- Sinarest 1-0-1 (10)
T- Montow-Lc 0-0-1 (10)
Susp- Grilinctus (10 ml) 1-1-1
T- Clanar 625 1-0-1 x 5 days
Cap- Haufy-PLX 0-0-1 (10)
Cap- Revgool-D 1-0-0 (10)
Pan-D BD before food
(Monfurate)
Nebulizer
Duolin + Budecort respules 1-1-1
Advice
Hot water gargles
"""

SAMPLE_PGX = """--- PAGE 1 ---
CONTENTS
I. PATIENT INFORMATION ........ 1
II. GENOTYPE RESULTS ........ 2
III. GENE DRUG INTERACTION SUMMARY ........ 3
IV. DIAGNOSIS SUMMARY ........ 4
V. TEST METHODOLOGY ........ 5
VI. PATIENT DRUG METABOLISM CARD ........ 6
I. PATIENT INFORMATION
Patient sample collected for pharmacogenomic genotyping panel.
II. GENOTYPE RESULTS
CYP2D6 *4/*4 PM [Low activity]
CYP2C19 *1/*17 RM [Increased activity]
CYP2C9 *1/*1 EM [Normal activity]
--- PAGE 2 ---
SLC6A4 L(A)/L(A) [High Activity]
HLA-A*31:01 Positive [Increased risk of skin reactions]
HLA-B*15:02 Negative [Low risk]
CYP3A4 *1/*1 EM [Normal activity] followed by a long description of hepatic enzymes.
CYP3A5 *3/*3 PM [Low activity]
CYP1A2 *1F/*1F Ultrarapid [High activity]
III. GENE DRUG INTERACTION SUMMARY
CYP2D6 *1/*1 codeine [Avoid]
IV. DIAGNOSIS SUMMARY
NUDT15 *1/*3 [Low activity]
V. TEST METHODOLOGY
TPMT *1/*3A assay list. Reference 247-252.
VI. PATIENT DRUG METABOLISM CARD
CYP2D6 *4/*4 PM [Low activity]
"""


@pytest.fixture(scope="module")
def payload():
    return rx.generate_rag_payload(SAMPLE_RX, SAMPLE_PGX, "test")


def _meds(payload):
    return payload["segregated_data"]["clinical_note"]["parsed_entities"]["medications"]


def _genes(payload):
    return {g["gene"]: g for g in payload["segregated_data"]["pgx_report"]["all_extracted_genotypes"]}


# --------------------------------------------------------------------------- output schema unchanged
def test_payload_schema_unchanged(payload):
    assert set(payload) == {
        "schema_version", "processing_metadata", "rag_instructions",
        "segregated_data", "rag_search_queries", "llm_context_prompt",
    }
    assert payload["schema_version"] == "rx-rag-dual-intake/2.0"
    assert set(payload["segregated_data"]["clinical_note"]["parsed_entities"]) == {
        "symptoms", "history", "medications", "investigations", "advice_and_procedures", "allergies",
    }
    for m in _meds(payload):
        assert set(m) == {
            "raw", "form", "written_name", "resolved_brand", "generic_composition", "therapeutic_class",
            "schedule", "doses_per_day", "quantity_or_volume", "confidence",
        }
    for g in _genes(payload).values():
        assert set(g) == {"gene", "category", "genotype", "phenotype", "actionable_alert", "evidence_context"}


# --------------------------------------------------------------------------- PGx
def test_sections_sliced_despite_table_of_contents(payload):
    genes = _genes(payload)
    assert "CYP2D6" in genes and genes["CYP2D6"]["genotype"] == "*4/*4"
    assert "NUDT15" not in genes  # Section IV
    assert "TPMT" not in genes  # Section V methodology


def test_slc6a4_parenthesised_diplotype(payload):
    assert _genes(payload)["SLC6A4"]["genotype"] == "L(A)/L(A)"


def test_rapid_and_ultrarapid_are_actionable(payload):
    genes = _genes(payload)
    assert genes["CYP2C19"]["phenotype"].startswith("Rapid Metabolizer (RM)")
    assert genes["CYP2C19"]["actionable_alert"] is True
    assert genes["CYP1A2"]["phenotype"].startswith("Ultrarapid Metabolizer (UM)")
    assert genes["CYP1A2"]["actionable_alert"] is True


def test_low_risk_and_normal_not_actionable(payload):
    genes = _genes(payload)
    assert genes["HLA-B*15:02"]["actionable_alert"] is False
    assert genes["CYP2C9"]["actionable_alert"] is False


def test_cyp3a4_does_not_inherit_cyp3a5_genotype(payload):
    genes = _genes(payload)
    assert genes["CYP3A4"]["genotype"] == "*1/*1"
    assert genes["CYP3A5"]["genotype"] == "*3/*3"


def test_section_v_inside_iv_heading_not_cut():
    text = "II. GENOTYPE RESULTS\nCYP2D6 *4/*4 PM [Low activity]\nIV. TEST METHODOLOGY OVERVIEW\nbody\n"
    assert "CYP2D6" in rx._clean_pgx_sections(text)


# --------------------------------------------------------------------------- RAG queries are patient-derived
def test_q3_uses_patient_results_only():
    pgx = "II. GENOTYPE RESULTS\nCYP2C19 *2/*2 PM [Low activity]\n"
    p = rx.generate_rag_payload("T- Sinarest 1-0-1", pgx, "test")
    q3 = next(q for q in p["rag_search_queries"] if q["id"] == "Q3_drug_gene_cross_check")["query"]
    assert "CYP2C19" in q3 and "*2/*2" in q3
    assert "CYP2D6" not in q3 and "*6/*6" not in q3


def test_no_invented_presentation():
    p = rx.generate_rag_payload("T- Sinarest 1-0-1", "", "test")
    assert "Inferred acute respiratory" not in p["llm_context_prompt"]
    assert "respiratory symptoms" not in p["rag_search_queries"][0]["query"]


# --------------------------------------------------------------------------- clinical note
def test_sample_medications_resolved(payload):
    got = [(m["resolved_brand"], m["schedule"]) for m in _meds(payload)]
    assert got == [
        ("Sinarest", "1-0-1"), ("Montair-LC", "0-0-1"), ("Grilinctus", "1-1-1"), ("Clavam 625", "1-0-1"),
        ("Hifenac-P", "0-0-1"), ("Razo-D", "1-0-0"), ("Pan-D", "BD"), ("Mometasone", None),
        ("Duolin", "1-1-1"), ("Budecort", "1-1-1"),
    ]
    clavam = _meds(payload)[3]
    assert clavam["written_name"] == "Clanar 625"  # duration stripped from the name


def test_demographics_and_vitals(payload):
    demo = payload["segregated_data"]["clinical_note"]["patient_demographics_and_vitals"]
    assert demo == {
        "name": "Ramesh Kumar", "age_years": 45, "sex": "Male", "date": "12/10/2026",
        "location": "Pune", "bp": "130/80", "temperature_f": 101.2,
    }


def test_word_prefixes_are_not_section_headers(payload):
    ents = payload["segregated_data"]["clinical_note"]["parsed_entities"]
    assert [s["raw"] for s in ents["symptoms"]] == ["40 Gark & Smafly x 5 days", "Cough since 3 days"]
    assert ents["history"] == []
    assert "Hot water gargles" in ents["advice_and_procedures"]
    assert ents["investigations"] == ["Chest X-ray", "Sputum for (AFB)"]


@pytest.mark.parametrize("line,section", [("Medrol 4mg 1-0-0", None), ("Testosterone", None), ("Cold since 2 days", None),
                                          ("Rx: T- Dolo 650", "meds"), ("c/o fever", "symptoms"), ("Inv - CBC", "investigations")])
def test_section_header_boundaries(line, section):
    hit = next((name for name, r in rx.SECTION_HEADER_RX if r.match(line)), None)
    assert hit == section


def test_bp_not_taken_from_date():
    assert "bp" not in rx.parse_demographics_and_vitals("Date: 12/10/2026")
    assert rx.parse_demographics_and_vitals("Date: 12/10/2026\n130/80 mmHg")["bp"] == "130/80"


@pytest.mark.parametrize("text,age,sex", [
    ("Age: 6 months", 0, None), ("Age: 18m", 1, None), ("Age/Sex: 30/M", 30, "Male"),
    ("Ramesh 45/F", 45, "Female"), ("Age: 45M", 45, None),
])
def test_age_units_and_combined_age_sex(text, age, sex):
    demo = rx.parse_demographics_and_vitals(text)
    assert demo.get("age_years") == age
    assert demo.get("sex") == sex


@pytest.mark.parametrize("raw,brand,generic_has", [
    ("Montair 10", "Montair", "Montelukast"),
    ("Montair LC", "Montair-LC", "Levocetirizine"),
    ("MontairLC", "Montair-LC", "Levocetirizine"),
    ("Grilinctus CD", "Grilinctus-CD", "Codeine"),
    ("Pantocid-D 40", "Pan-D", "Pantoprazole"),
    ("Omez-D", "Omez-D", "Omeprazole"),
    ("Clavam 375", "Clavam 375", "250mg"),
    ("Sinarest LP", "Sinarest-LP", "Levocetirizine"),
    ("Montoir-LC", "Montair-LC", "Levocetirizine"),  # fuzzy OCR misread
])
def test_brand_resolution(raw, brand, generic_has):
    canon, generic, _, conf = rx.resolve_indian_brand(raw)
    assert canon == brand and generic_has in generic and conf >= 0.72


def test_unrelated_words_not_fuzzy_matched():
    assert rx.resolve_indian_brand("Ascorbic acid")[1] is None
    assert rx.resolve_indian_brand("pain")[1] is None


@pytest.mark.parametrize("line,schedule,per_day", [
    ("T. Dolo 650 1-1-1-1", "1-1-1-1", 4), ("T. Dolo 650 ½-0-½", "½-0-½", 1), ("Syr. Ascoril LS 5 ml TDS", "TDS", 3),
])
def test_schedules(line, schedule, per_day):
    import unicodedata
    m = rx.parse_med_line(unicodedata.normalize("NFKC", line))
    assert (m["schedule"], m["doses_per_day"]) == (schedule, per_day)


def test_syrup_prefix_form():
    m = rx.parse_med_line("Syr. Ascoril LS 5 ml TDS")
    assert m["form"] == "Syrup" and m["resolved_brand"] == "Ascoril-LS"


def test_negated_cough_not_diagnosed():
    ents = rx.parse_clinical_note("c/o fever, no cough")["parsed_entities"]
    assert "Upper Respiratory" not in ents["symptoms"][0]["clinical_interpretation"]


def test_blood_sugar_history_stays_history():
    ents = rx.parse_clinical_note("h/o high blood sugar since 5 yrs\nInv: FBS, PPBS")["parsed_entities"]
    assert ents["history"][0]["raw"] == "high blood sugar since 5 yrs"
    assert ents["investigations"] == ["FBS, PPBS"]


def test_symptoms_on_demographic_line_kept():
    ents = rx.parse_clinical_note("Age: 30 c/o fever x 3 days")["parsed_entities"]
    assert [s["raw"] for s in ents["symptoms"]] == ["fever x 3 days"]


def test_unprefixed_drug_under_rx_is_medication():
    ents = rx.parse_clinical_note("Rx\nMontair LC\nSteam inhalation")["parsed_entities"]
    assert [m["resolved_brand"] for m in ents["medications"]] == ["Montair-LC"]
    assert ents["advice_and_procedures"] == ["Steam inhalation"]
    assert not ents["symptoms"]


def test_nkda_recorded():
    assert rx.parse_clinical_note("NKDA")["parsed_entities"]["allergies"] == ["NKDA (no known drug allergies)"]


def test_scanned_pdf_reports_clear_error(tmp_path):
    from pypdf import PdfWriter

    pdf = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    with open(pdf, "wb") as fh:
        w.write(fh)
    with pytest.raises(RuntimeError, match=re.escape("no extractable text")):
        rx.parse_pdf(pdf)
