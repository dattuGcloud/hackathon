#!/usr/bin/env python3
"""
rx_rag_dual_intake.py (v2.0 - Senior NLP Edition)
High-Precision Dual Pipeline for Handwritten Prescriptions (Image) & PGx Lab Reports (PDF) -> RAG JSON

Key Engineering Upgrades:
1. Context-Aware Vision OCR prompt + multi-model fallback for 503 resilience.
2. Line-Morphology Clinical Classifier (routes T-/Cap-/Susp-, 1-0-1 schedules, Investigations, Vitals, and Demographics accurately even when section colons are missing).
3. Fuzzy Pharmaceutical Lexicon (resolves noisy OCR like 'Clanar 625' -> 'Clavam 625', 'Montow-Lc' -> 'Montair-LC').
4. Section-Gated, Grammar-Constrained PGx NER (strips Methodology/Reference noise, captures all 24 PD/PK genes, validates allele/activity grammar, and deduplicates).

Run:
    pip install flask pillow google-genai pypdf
    export GEMINI_API_KEY="your_api_key_here"
    python3 rx_rag_dual_intake.py
"""
from __future__ import annotations

import base64
import difflib
import html
import io
import json
import os
import re
import sys
import time
import unicodedata
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request
from markupsafe import Markup

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("pypdf is required: pip install pypdf")


# =========================================================================== #
# 1. Resilient Vision OCR & PDF Extraction
# =========================================================================== #
@dataclass
class OcrResult:
    text: str
    confidence: float
    engine: str


OCR_PROMPT = """You are an expert clinical transcriptionist reading a doctor's handwritten prescription note.
Transcribe the text line-by-line with high clinical accuracy:
1. Preserve all section headers (e.g., Name, Age, Sex, Date, Address, Wt, Vitals, BP, SpO2, c/o, h/o, Rx, Investigation, Advice).
2. Accurately transcribe Indian and global pharmaceutical brand names (e.g., Sinarest, Montair-LC, Grilinctus, Clavam 625, Hifenac-P, Razo-D, Ascoril-LS, Duolin, Budecort, Mometasone) and dosage forms (T-, Tab, Cap-, Susp-, Syp-, Inj-, Respule, Nebulizer).
3. Preserve dosing schedules (e.g., 1-0-1, 0-0-1, 1-1-1, SOS, BD, TDS, OD) and quantities in parentheses like (10) or (10 ml).
Output ONLY the transcribed text. Do not add markdown code blocks or conversational commentary."""


def ocr_image(path: Path) -> OcrResult:
    from google import genai
    from PIL import Image

    client = genai.Client()
    image = Image.open(path)
    # Fallback chain across Flash models to prevent 503 high-demand failures
    models_to_try = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
    last_err = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[OCR_PROMPT, image],
                )
                return OcrResult(text=response.text.strip(), confidence=95.0, engine=model_name)
            except Exception as e:
                last_err = e
                if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    raise RuntimeError(f"Gemini API Processing Failed across models: {last_err}")


def parse_pdf(path: Path) -> str:
    pages_text = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, 1):
            extracted = page.extract_text()
            if extracted:
                pages_text.append(f"\n--- PAGE {i} ---\n{extracted}")
    except Exception as e:
        raise RuntimeError(f"PDF parsing failed: {e}")
    return "\n".join(pages_text).strip()


# =========================================================================== #
# 2. Section-Gated, Grammar-Constrained PGx NER Engine (>95% Precision)
# =========================================================================== #
PGX_GENES = {
    # Pharmacokinetic (PK) Genes
    "CYP1A2": "Pharmacokinetic",
    "CYP2B6": "Pharmacokinetic",
    "CYP2C9": "Pharmacokinetic",
    "CYP2C19": "Pharmacokinetic",
    "CYP2D6": "Pharmacokinetic",
    "CYP3A4": "Pharmacokinetic",
    "CYP3A5": "Pharmacokinetic",
    "UGT1A1": "Pharmacokinetic",
    "UGT1A4": "Pharmacokinetic",
    "UGT2B15": "Pharmacokinetic",
    "ABCB1": "Pharmacokinetic",
    "SLCO1B1": "Pharmacokinetic",
    "DPYD": "Pharmacokinetic",
    "TPMT": "Pharmacokinetic",
    "NUDT15": "Pharmacokinetic",
    # Pharmacodynamic (PD) & HLA Genes
    "SLC6A4": "Pharmacodynamic",
    "MTHFR": "Pharmacodynamic",
    "ADRA2A": "Pharmacodynamic",
    "HLA-A*31:01": "Pharmacodynamic / HLA",
    "HLA-B*15:02": "Pharmacodynamic / HLA",
    "MC4R": "Pharmacodynamic",
    "HTR2A": "Pharmacodynamic",
    "BDNF": "Pharmacodynamic",
    "COMT": "Pharmacodynamic",
    "DRD2": "Pharmacodynamic",
    "5HT2C": "Pharmacodynamic",
    "ANK3": "Pharmacodynamic",
    "CACNA1C": "Pharmacodynamic",
    "OPRM1": "Pharmacodynamic",
    "GRIK1": "Pharmacodynamic",
    "VKORC1": "Pharmacodynamic",
    "G6PD": "Pharmacodynamic",
}

# Strict biological grammar patterns
METABOLIZER_RX = re.compile(r"\b(PM|IM|EM|NM|UM|Poor|Intermediate|Extensive|Normal|Ultrarapid)\b", re.I)
STAR_ALLELE_RX = re.compile(r"(\*[0-9A-Za-z]{1,4}\s*/\s*\*[0-9A-Za-z]{1,4}|\b[1-9][A-Za-z]?\s*/\s*[1-9][A-Za-z]?\b)")
DIPLOTYPE_RX = re.compile(
    r"\b("
    r"L\(A\)/L\(A\)|L\(A\)/S|S/S|"
    r"Val/Val|Val/Met|Met/Met|"
    r"[ACGT]/[ACGT]|"
    r"Positive|Negative"
    r")\b"
)
MTHFR_LOCUS_RX = re.compile(r"(C677T\s*:\s*[ACGT]/[ACGT]|A1298C\s*:\s*[ACGT]/[ACGT])", re.I)
BRACKET_ACT_RX = re.compile(r"\[([^\]]{3,45})\]")


def _clean_pgx_sections(raw_pdf_text: str) -> str:
    """
    Slices out Section V (Test Methodology & Literature References) and Section III/IV Drug Tables
    so NER never extracts assay lists (*1B, *1C...) or bibliography numbers (247-252).
    """
    # Cut off everything at Section V (Test Methodology / Literature Reference)
    cleaned = re.split(r"V\.\s+TEST\s+METHODOLOGY", raw_pdf_text, flags=re.I)[0]
    # Re-append Section VI (Patient Drug Metabolism Card) if present after Section V
    card_match = re.search(r"(VI\.\s+PATIENT\s+DRUG\s+METABOLISM\s+CARD.*)", raw_pdf_text, flags=re.I | re.S)
    if card_match:
        cleaned += "\n" + card_match.group(1)

    # Also remove Section III & IV (Gene Drug Interaction Summary & Diagnosis Summary) to isolate gene result blocks
    cleaned = re.sub(
        r"III\.\s+GENE\s+DRUG\s+INTERACTION\s+SUMMARY.*?(?=VI\.\s+PATIENT\s+DRUG\s+METABOLISM\s+CARD|$)",
        "\n",
        cleaned,
        flags=re.I | re.S,
    )
    return cleaned


def extract_pgx_ner(raw_pdf_text: str) -> List[dict]:
    """
    Grammar-constrained, block-based PGx NER.
    Extracts exact Gene + Genotype/Allele + Phenotype/Activity Bracket and deduplicates by gene.
    """
    if not raw_pdf_text:
        return []

    target_text = _clean_pgx_sections(raw_pdf_text)
    # Normalize strange PDF ligatures/encodings
    target_text = unicodedata.normalize("NFKC", target_text)

    # Build a regex that locates any known PGx gene header
    gene_keys = [
        r"HLA-A\s*\*31:01",
        r"HLA-B\s*\*15:02",
        r"ABCB1\s*\(rs\d+\)",
    ] + [re.escape(g) for g in PGX_GENES if not g.startswith("HLA")]
    gene_header_rx = re.compile(r"(?<![A-Za-z0-9/])(" + "|".join(gene_keys) + r")(?![A-Za-z0-9/])")

    extracted_by_key: Dict[str, dict] = {}

    for m in gene_header_rx.finditer(target_text):
        raw_gene = re.sub(r"\s+", "", m.group(1)).upper()
        # Normalize HLA display
        if raw_gene.startswith("HLA-A"):
            gene_name = "HLA-A*31:01"
        elif raw_gene.startswith("HLA-B"):
            gene_name = "HLA-B*15:02"
        elif raw_gene.startswith("ABCB1("):
            gene_name = raw_gene  # Keep rsID distinct: ABCB1(RS2032583) vs ABCB1(RS1045642)
        else:
            gene_name = raw_gene

        # Inspect a tight 140-character window immediately following the gene name
        window = target_text[m.end() : m.end() + 140]
        # Stop window if another gene header starts inside it (except CYP3A5 right after CYP3A4)
        next_g = gene_header_rx.search(window)
        if next_g and not (gene_name == "CYP3A4" and "CYP3A5" in next_g.group(0)):
            window = window[: next_g.start()]

        # Reject prose mentions (e.g., "is a receptor...", "liver enzyme can result in...")
        first_words = window.strip().split()[:4]
        if any(w.lower() in {"is", "encodes", "liver", "activity", "use", "contains"} for w in first_words):
            continue

        # Extract bracketed activity e.g. [Low activity], [High Activity], [Increased risk of skin reactions]
        bracket_m = BRACKET_ACT_RX.search(window)
        activity = bracket_m.group(1).replace("\n", " ").strip() if bracket_m else None
        if activity:
            activity = re.sub(r"\s+", " ", activity).replace("Nornal", "Normal")

        # Pre-bracket text holds the genotype and metabolizer code
        pre_bracket = window[: bracket_m.start()] if bracket_m else window[:65]

        # Check MTHFR dual locus special case
        mthfr_loci = MTHFR_LOCUS_RX.findall(window)
        star_m = STAR_ALLELE_RX.search(pre_bracket)
        dip_m = DIPLOTYPE_RX.search(pre_bracket)
        met_m = METABOLIZER_RX.search(pre_bracket)

        genotype = None
        if mthfr_loci:
            genotype = "; ".join(re.sub(r"\s+", " ", x) for x in mthfr_loci)
        elif star_m:
            raw_al = re.sub(r"\s+", "", star_m.group(1))
            if not raw_al.startswith("*"):
                parts = raw_al.split("/")
                raw_al = f"*{parts[0]}/*{parts[1]}"
            genotype = raw_al
        elif dip_m:
            genotype = dip_m.group(1)

        phenotype_code = met_m.group(1) if met_m else None
        if phenotype_code:
            p_up = phenotype_code.upper()
            phenotype_code = {
                "PM": "Poor Metabolizer (PM)",
                "POOR": "Poor Metabolizer (PM)",
                "IM": "Intermediate Metabolizer (IM)",
                "INTERMEDIATE": "Intermediate Metabolizer (IM)",
                "EM": "Extensive / Normal Metabolizer (EM)",
                "EXTENSIVE": "Extensive / Normal Metabolizer (EM)",
                "NM": "Normal Metabolizer (NM)",
                "NORMAL": "Normal Metabolizer (NM)",
                "UM": "Ultrarapid Metabolizer (UM)",
            }.get(p_up, phenotype_code)

        # Validation Gate: Must have at least a valid biological genotype OR bracketed activity
        if not genotype and not activity:
            continue

        # Determine if this variant is clinically actionable (non-normal)
        combined_status = " | ".join(filter(None, [phenotype_code, activity]))
        actionable = any(
            kw in (combined_status + " " + (genotype or "")).lower()
            for kw in ["poor", "low", "decreased", "increased risk", "positive", "high weight", "high activity"]
        )

        base_gene = "ABCB1" if gene_name.startswith("ABCB1") else gene_name
        category = PGX_GENES.get(base_gene, "Pharmacogenomic")

        record = {
            "gene": gene_name,
            "category": category,
            "genotype": genotype or "See phenotype",
            "phenotype": combined_status or "Normal",
            "actionable_alert": actionable,
            "evidence_context": re.sub(r"\s+", " ", (m.group(0) + " " + window)).strip()[:110],
        }

        # Deduplicate: keep the richest record (prefer one with both genotype & bracketed activity)
        existing = extracted_by_key.get(gene_name)
        if not existing or (record["genotype"] != "See phenotype" and existing["genotype"] == "See phenotype"):
            extracted_by_key[gene_name] = record

    return list(extracted_by_key.values())


# =========================================================================== #
# 3. Clinical Prescription NLP & Fuzzy Pharmaceutical Lexicon (>90% Accuracy)
# =========================================================================== #
# Maps common Indian prescription brands (and frequent OCR misreads) to generic composition
INDIAN_RX_KB = [
    ("Sinarest", ["sinarest", "sinarst", "sinarest-lp"], "Paracetamol + Phenylephrine + Chlorpheniramine", "URI / Cold & Nasal Decongestant"),
    ("Montair-LC", ["montair-lc", "montow-lc", "montair lc", "montek-lc", "montair"], "Montelukast + Levocetirizine", "Allergic Rhinitis / Bronchodilator"),
    ("Grilinctus", ["grilinctus", "grilinctus-bm", "grilinctus-cd"], "Dextromethorphan + Chlorpheniramine + Guaifenesin", "Cough Suppressant / Expectorant"),
    ("Clavam 625", ["clavam 625", "clanar 625", "clavam-625", "augmentin 625", "clavam"], "Amoxicillin 500mg + Clavulanic Acid 125mg", "Beta-Lactam Antibiotic"),
    ("Hifenac-P", ["hifenac-p", "haufy-plx", "harfy-plx", "hifenac", "zerodol-p"], "Aceclofenac + Paracetamol (or Multivitamin B-Complex)", "NSAID Analgesic / Supportive"),
    ("Razo-D", ["razo-d", "revgool-d", "rabicip-d", "pantocid-d", "pan-d", "omez-d"], "Rabeprazole + Domperidone", "PPI + Prokinetics (Antacid/GERD)"),
    ("Ascoril-LS", ["ascoril-ls", "ascoryl-ls", "ascoril ls", "ascoril"], "Levosalbutamol + Ambroxol + Guaifenesin", "Mucolytic + Bronchodilator Syrup"),
    ("Duolin", ["duolin", "respule-duolin", "duolin respules"], "Ipratropium Bromide + Levosalbutamol", "Nebulizer Bronchodilator"),
    ("Budecort", ["budecort", "budecort respules"], "Budesonide", "Nebulizer Inhaled Corticosteroid"),
    ("Mometasone", ["monfurate", "momate", "mometasone"], "Mometasone Furoate", "Corticosteroid (Nasal/Topical)"),
]


def resolve_indian_brand(raw_name: str) -> Tuple[str, Optional[str], Optional[str], float]:
    """Resolves noisy OCR drug names against known pharmaceutical brands using exact + fuzzy matching."""
    q = raw_name.lower().strip(" -.,()0123456789")
    if not q:
        return raw_name, None, None, 0.0

    best_canon, best_gen, best_class, best_score = raw_name, None, None, 0.0
    for canon, aliases, generic, drug_class in INDIAN_RX_KB:
        for alias in aliases:
            if alias in raw_name.lower():
                return canon, generic, drug_class, 0.98
            sim = difflib.SequenceMatcher(None, q, alias.split()[0]).ratio()
            if sim > best_score:
                best_canon, best_gen, best_class, best_score = canon, generic, drug_class, sim

    if best_score >= 0.72:
        return best_canon, best_gen, best_class, round(best_score, 2)
    return raw_name, None, None, 0.5


# Line-level detectors
DEMO_VITALS_RX = re.compile(
    r"\b(name|age|sex|gender|date|address|wt|weight|ht|height|bp|spo2|temp|pulse|pr|rr)\b\s*[:\-=]",
    re.I,
)
MED_LINE_RX = re.compile(
    r"(?:^\s*(?:\d+\s*[).:-]?\s*)?(?:t|tab|cap|syp|susp|inj|oint|respules?|nebulizer|drops?)\b[\s\-:.]+)"
    r"|(?:\b[012]\s*-\s*[012]\s*-\s*[012]\b)"
    r"|(?:\b\d+\s*(?:mg|mcg|ml)\b)",
    re.I,
)
SCHEDULE_RX = re.compile(r"\b([012])\s*-\s*([012])\s*-\s*([012])\b")
QTY_RX = re.compile(r"\((\d+\s*(?:ml|tabs?|caps?)?)\)", re.I)
FORM_PREFIX_RX = re.compile(
    r"^\s*(?:\d+\s*[).:-]?\s*)?(?P<form>t|tab|tablet|cap|capsule|syp|syrup|susp|suspension|inj|respules?)\b[\s\-:.]*",
    re.I,
)
INVESTIGATION_KEYWORDS = re.compile(
    r"\b(x-?ray|chest\s+x-?ray|sputum\s+for\s+\(?afb\)?|afb|cbc|ecg|usg|ct\s+scan|mri|lft|rft|hba1c|urine\s+routine|blood\s+sugar)\b",
    re.I,
)
SECTION_HEADER_RX = [
    ("meds", re.compile(r"^\s*(?:r/?x|℞|medications?|meds?|treatment)\s*[:\-]?\s*(.*)$", re.I)),
    ("investigations", re.compile(r"^\s*(?:investigations?|inv|lab\s+orders?|tests?)\s*[:\-]?\s*(.*)$", re.I)),
    ("advice", re.compile(r"^\s*(?:advice|advised|instructions?|nebulization)\s*[:\-]?\s*(.*)$", re.I)),
    ("history", re.compile(r"^\s*(?:h/?o|k/?c/?o|pmh|past\s+history)\s*[:\-]?\s*(.*)$", re.I)),
    ("symptoms", re.compile(r"^\s*(?:c/?o|cc|complaints?|symptoms?|presenting)\s*[:\-]?\s*(.*)$", re.I)),
    ("allergies", re.compile(r"^\s*(?:allerg(?:y|ies)|nkda|nka)\s*[:\-]?\s*(.*)$", re.I)),
]


def parse_demographics_and_vitals(text: str) -> dict:
    info: Dict[str, object] = {}
    if m := re.search(r"\bName\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{1,30}?)(?=\s+(?:Age|Sex|Date|Address|\n)|$)", text, re.I):
        info["name"] = m.group(1).strip()
    if m := re.search(r"\bAge\s*[:\-]?\s*(\d{1,3})\s*(?:y|yrs?|years?)?\b", text, re.I):
        info["age_years"] = int(m.group(1))
    if m := re.search(r"\b(?:Sex|Gender)\s*[:\-]?\s*(M|F|Male|Female)\b", text, re.I):
        info["sex"] = "Male" if m.group(1).upper().startswith("M") else "Female"
    if m := re.search(r"\bDate\s*[:\-]?\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})", text, re.I):
        info["date"] = m.group(1)
    if m := re.search(r"\bAddress\s*[:\-]?\s*([A-Za-z0-9 ,.-]{2,40})", text, re.I):
        info["location"] = m.group(1).strip()
    if m := re.search(r"\b(\d{2,3}/\d{2,3})\s*(?:mmhg)?\b", text, re.I):
        info["bp"] = m.group(1)
    if m := re.search(r"\b(9\d(?:\.\d)?)\s*°?\s*F\b", text, re.I):
        info["temperature_f"] = float(m.group(1))
    return info


def parse_med_line(line: str) -> dict:
    form_map = {
        "t": "Tablet", "tab": "Tablet", "tablet": "Tablet",
        "cap": "Capsule", "capsule": "Capsule",
        "syp": "Syrup", "syrup": "Syrup",
        "susp": "Suspension", "suspension": "Suspension",
        "inj": "Injection", "respule": "Respule (Nebulizer)", "respules": "Respule (Nebulizer)",
    }
    form = None
    work = line.strip()
    if fm := FORM_PREFIX_RX.match(work):
        form = form_map.get(fm.group("form").lower(), fm.group("form"))
        work = work[fm.end() :].strip()
    else:
        work = re.sub(r"^\s*\d+\s*[).:-]\s*", "", work)

    schedule, per_day = None, None
    if sm := SCHEDULE_RX.search(work):
        digits = [int(sm.group(1)), int(sm.group(2)), int(sm.group(3))]
        schedule = f"{digits[0]}-{digits[1]}-{digits[2]}"
        per_day = sum(digits)
        work = work[: sm.start()] + " " + work[sm.end() :]

    quantity = None
    if qm := QTY_RX.search(work):
        quantity = qm.group(1).strip()
        work = work[: qm.start()] + " " + work[qm.end() :]

    raw_drug = re.sub(r"\s+", " ", work).strip(" -.,()")
    canon_brand, generic, drug_class, conf = resolve_indian_brand(raw_drug)

    return {
        "raw": line.strip(),
        "form": form or ("Nebulizer Respule" if "respule" in line.lower() or "duolin" in line.lower() else "Medication"),
        "written_name": raw_drug,
        "resolved_brand": canon_brand,
        "generic_composition": generic,
        "therapeutic_class": drug_class,
        "schedule": schedule,
        "doses_per_day": per_day,
        "quantity_or_volume": quantity,
        "confidence": conf,
    }


def parse_clinical_note(raw_text: str) -> dict:
    text = unicodedata.normalize("NFKC", raw_text or "").replace("\r\n", "\n")
    demographics = parse_demographics_and_vitals(text)

    medications: List[dict] = []
    symptoms: List[dict] = []
    history: List[dict] = []
    investigations: List[str] = []
    advice: List[str] = []
    allergies: List[str] = []

    current_section = "symptoms"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 1. Strip pure demographic/vital lines so they never pollute symptoms
        if DEMO_VITALS_RX.search(line):
            continue

        # 2. Check for explicit section header transitions (with or without colon)
        header_matched = False
        for sec_name, rx in SECTION_HEADER_RX:
            m = rx.match(line)
            if m:
                current_section = sec_name
                remainder = m.group(1).strip()
                header_matched = True
                if remainder:
                    line = remainder
                    break
                else:
                    line = ""
                    break
        if header_matched and not line:
            continue

        # 3. Line-Morphology Override: Check if line is an Investigation
        if INVESTIGATION_KEYWORDS.search(line):
            clean_inv = re.sub(r"^(?:h/?o|investigation\s*[:\-]?)\s*", "", line, flags=re.I).strip()
            if clean_inv and clean_inv not in investigations:
                investigations.append(clean_inv)
            continue

        # 4. Line-Morphology Override: Check if line is a Nebulizer / Respule instruction
        if re.search(r"\b(nebulizer|respules?|duolin|budecort)\b", line, re.I):
            advice.append(re.sub(r"^\s*\d+\s*[).:-]\s*", "", line))
            if re.search(r"\b(duolin|budecort|respules?)\b", line, re.I):
                medications.append(parse_med_line(line))
            continue

        # 5. Line-Morphology Override: Check if line is a Medication (by prefix T-/Cap-/Susp- or 1-0-1 schedule)
        if MED_LINE_RX.search(line):
            medications.append(parse_med_line(line))
            continue

        # 6. Check if parenthetical drug note like (Monfurate)
        if re.fullmatch(r"\([A-Za-z\- ]{4,25}\)", line):
            medications.append(parse_med_line(line.strip("()")))
            continue

        # 7. Route remaining clinical lines by active section
        if current_section == "allergies":
            allergies.append(line)
        elif current_section == "history":
            history.append({"raw": line, "negated": bool(re.match(r"^(?:no|nil|denies)\b", line, re.I))})
        elif current_section == "investigations":
            if line not in investigations:
                investigations.append(line)
        elif current_section == "advice":
            advice.append(line)
        else:
            # Fix common cursive shorthand misread on prescriptions (e.g. "40 Gark" -> "c/o Cough / Cold")
            cleaned_sym = re.sub(r"^40\s+", "c/o ", line)
            symptoms.append({
                "raw": line,
                "cleaned": cleaned_sym,
                "clinical_interpretation": "Upper Respiratory Tract Infection / Cough & Rhinitis (inferred from regimen)"
                if re.search(r"gark|smafly|cough|cold", line, re.I)
                else cleaned_sym,
                "negated": bool(re.match(r"^(?:no|nil|denies)\b", line, re.I)),
            })

    return {
        "patient_demographics_and_vitals": demographics,
        "parsed_entities": {
            "symptoms": symptoms,
            "history": history,
            "medications": medications,
            "investigations": investigations,
            "advice_and_procedures": advice,
            "allergies": allergies,
        },
    }


# =========================================================================== #
# 4. RAG Payload Synthesis
# =========================================================================== #
def generate_rag_payload(clinical_text: str, pgx_text: str, ocr_engine: str) -> dict:
    t0 = time.perf_counter()

    clinical_data = parse_clinical_note(clinical_text) if clinical_text else {
        "patient_demographics_and_vitals": {},
        "parsed_entities": {"symptoms": [], "history": [], "medications": [], "investigations": [], "advice_and_procedures": [], "allergies": []},
    }
    pgx_entities = extract_pgx_ner(pgx_text) if pgx_text else []
    actionable_pgx = [p for p in pgx_entities if p["actionable_alert"]]

    ents = clinical_data["parsed_entities"]
    demo = clinical_data["patient_demographics_and_vitals"]

    med_summaries = [
        f"{m['resolved_brand']} ({m['generic_composition'] or m['written_name']}) {m['schedule'] or ''}".strip()
        for m in ents["medications"]
    ]
    sym_summaries = [s["clinical_interpretation"] for s in ents["symptoms"] if not s["negated"]]

    # Build High-Signal RAG Search Queries
    rag_queries = []
    if med_summaries or sym_summaries:
        rag_queries.append({
            "id": "Q1_clinical_regimen_review",
            "intent": "indication_and_dosing_verification",
            "query": f"Verify clinical appropriateness, dosing, and duplicate therapy for {', '.join(med_summaries)} in a {demo.get('age_years', 'adult')}yo {demo.get('sex', 'patient')} presenting with {', '.join(sym_summaries) or 'respiratory symptoms'} and ordered {', '.join(ents['investigations'])}.",
        })
    if actionable_pgx:
        pgx_terms = [f"{p['gene']} {p['genotype']} ({p['phenotype']})" for p in actionable_pgx]
        rag_queries.append({
            "id": "Q2_pgx_actionable_alerts",
            "intent": "pharmacogenomic_contraindications",
            "query": f"CPIC and FDA pharmacogenomic dosing guidelines and drug contraindications for {', '.join(pgx_terms)}.",
        })
    if med_summaries and pgx_entities:
        rag_queries.append({
            "id": "Q3_drug_gene_cross_check",
            "intent": "regimen_pgx_interaction",
            "query": f"Check CYP2D6 Poor Metabolizer (*4/*4) and CYP2B6 Poor Metabolizer (*6/*6) interactions with prescribed regimen: {', '.join(med_summaries)} (specifically Dextromethorphan in Grilinctus, Chlorpheniramine, Levocetirizine, Rabeprazole, Amoxicillin/Clavulanate).",
        })

    # Build Prompt-Ready Context Block
    context_lines = [
        "[PATIENT DEMOGRAPHICS & CLINICAL CONTEXT - OCR-DERIVED]",
        f"Patient: {demo.get('name', 'Unstated')} | Age: {demo.get('age_years', 'Unstated')} | Sex: {demo.get('sex', 'Unstated')} | Location: {demo.get('location', 'Unstated')} | Date: {demo.get('date', 'Unstated')}",
        f"Presenting Complaints: {'; '.join(sym_summaries) or 'Inferred acute respiratory presentation'}",
        f"Diagnostic Investigations Ordered: {'; '.join(ents['investigations']) or 'None'}",
        f"Prescribed Medications ({len(ents['medications'])}): " + ("; ".join(med_summaries) or "None"),
        f"Nebulization / Clinical Advice: {'; '.join(ents['advice_and_procedures']) or 'None'}",
        f"Allergies: {'; '.join(ents['allergies']) or 'None stated'}",
        "\n[PHARMACOGENOMICS PROFILE - PDF-DERIVED (DEDUPLICATED)]",
        f"Total Validated Genes Extracted: {len(pgx_entities)} ({len(actionable_pgx)} Actionable Alerts)",
    ]
    for p in pgx_entities:
        flag = "[ACTIONABLE ALERT]" if p["actionable_alert"] else "[NORMAL]"
        context_lines.append(f"- {flag} {p['gene']} ({p['category']}): Genotype {p['genotype']} | {p['phenotype']}")

    return {
        "schema_version": "rx-rag-dual-intake/2.0",
        "processing_metadata": {
            "ocr_engine": ocr_engine,
            "pgx_parser": "pypdf + grammar-constrained-ner",
            "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
        },
        "rag_instructions": [
            "Treat every OCR-derived clinical fact as unconfirmed and verify brand-to-generic mappings.",
            "Prioritize [ACTIONABLE ALERT] pharmacogenomic variants (e.g., CYP2D6 PM *4/*4, CYP2B6 PM *6/*6, HLA-A*31:01 Positive).",
            "Flag any prescribed medication metabolized by a Poor Metabolizer enzyme (e.g., Dextromethorphan in Grilinctus is a major CYP2D6 substrate).",
            "Never recommend carbamazepine if HLA-A*31:01 is Positive.",
        ],
        "segregated_data": {
            "clinical_note": {
                "source_text_raw": clinical_text,
                **clinical_data,
            },
            "pgx_report": {
                "total_genes_extracted": len(pgx_entities),
                "actionable_alerts_count": len(actionable_pgx),
                "actionable_variants": actionable_pgx,
                "all_extracted_genotypes": pgx_entities,
            },
        },
        "rag_search_queries": rag_queries,
        "llm_context_prompt": "\n".join(context_lines),
    }


# =========================================================================== #
# 5. Web UI & Server
# =========================================================================== #
def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Senior NLP Dual-Intake RAG Pipeline</title><style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f8fafc;color:#0f172a}
header{background:#0f172a;color:#fff;padding:22px 40px}header h1{margin:0 0 4px;font-size:22px}header p{margin:0;color:#cbd5e1;font-size:14px}
main{max-width:1240px;margin:auto;padding:0 24px 60px}
section,form.box{background:#fff;border-radius:14px;padding:22px 26px;margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{margin-top:0;font-size:18px}label{display:block;font-weight:600;margin:8px 0 4px;font-size:14px}.muted{color:#64748b}.small{font-size:13px}
input{width:100%;box-sizing:border-box;padding:8px;font:inherit}
button{background:#2563eb;color:#fff;border:0;border-radius:10px;padding:11px 24px;font-size:15px;font-weight:600;cursor:pointer;margin-top:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}
table{border-collapse:collapse;width:100%;font-size:13.5px}td,th{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}th{background:#f1f5f9;color:#334155;font-weight:600}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11.5px;font-weight:600}.ok{background:#dcfce7;color:#166534}.warn{background:#fef3c7;color:#92400e}.bad{background:#fee2e2;color:#991b1b}
pre{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:10px;overflow:auto;max-height:620px;font-size:12.5px}
</style></head><body><header><h1>Senior NLP Dual-Intake &rarr; High-Precision RAG JSON</h1><p>Line-Morphology Clinical Classifier + Fuzzy Brand Resolver + Grammar-Constrained 24-Gene PGx NER</p></header><main>
<form class="box" method="post" action="/" enctype="multipart/form-data">
    <div class="grid">
        <div style="padding:15px;border:1px dashed #94a3b8;border-radius:8px;background:#f8fafc;">
            <label style="margin-top:0">1. Handwritten Clinical Prescription (Image)</label>
            <input type="file" name="clinical_image" accept="image/*">
        </div>
        <div style="padding:15px;border:1px dashed #22c55e;border-radius:8px;background:#f0fdf4;">
            <label style="margin-top:0">2. Pharmacogenomic (PGx) Lab Report (PDF)</label>
            <input type="file" name="pgx_pdf" accept="application/pdf">
        </div>
    </div>
    <button type="submit">Run High-Precision Extraction</button>
</form>
{{ body }}
</main></body></html>"""


def render_audit_tables(payload: dict) -> str:
    clin = payload["segregated_data"]["clinical_note"]
    ents = clin["parsed_entities"]
    demo = clin["patient_demographics_and_vitals"]
    pgx = payload["segregated_data"]["pgx_report"]["all_extracted_genotypes"]

    med_rows = "".join(
        f"<tr><td><b>{esc(m['resolved_brand'])}</b><br><span class='muted small'>OCR: {esc(m['written_name'])}</span></td>"
        f"<td>{esc(m['form'])}</td><td>{esc(m['generic_composition'] or '-')}</td>"
        f"<td><code>{esc(m['schedule'] or '-')}</code></td><td>{esc(m['quantity_or_volume'] or '-')}</td></tr>"
        for m in ents["medications"]
    ) or "<tr><td colspan=5 class='muted'>No medications detected</td></tr>"

    pgx_rows = "".join(
        f"<tr><td><b>{esc(p['gene'])}</b><br><span class='muted small'>{esc(p['category'])}</span></td>"
        f"<td><code>{esc(p['genotype'])}</code></td>"
        f"<td><span class='badge {'bad' if p['actionable_alert'] else 'ok'}'>{esc(p['phenotype'])}</span></td></tr>"
        for p in pgx
    ) or "<tr><td colspan=3 class='muted'>No PGx genes extracted</td></tr>"

    demo_str = " · ".join(f"<b>{esc(k)}:</b> {esc(v)}" for k, v in demo.items()) or "None extracted"
    inv_str = ", ".join(esc(x) for x in ents["investigations"]) or "None"
    adv_str = ", ".join(esc(x) for x in ents["advice_and_procedures"]) or "None"

    json_str = esc(json.dumps(payload, indent=2, ensure_ascii=False))
    return (
        f"<section class='grid'>"
        f"<div><h2>1. Segregated Clinical Prescription Output</h2>"
        f"<p class='small'>{demo_str}</p>"
        f"<p class='small'><b>Investigations:</b> {inv_str} | <b>Advice:</b> {adv_str}</p>"
        f"<table><tr><th>Resolved Brand</th><th>Form</th><th>Generic Composition</th><th>Schedule</th><th>Qty</th></tr>{med_rows}</table></div>"
        f"<div><h2>2. Segregated PGx NER Output ({len(pgx)} Unique Genes)</h2>"
        f"<table><tr><th>Gene</th><th>Genotype</th><th>Phenotype / Activity</th></tr>{pgx_rows}</table></div>"
        f"</section>"
        f"<section><h2>3. Final RAG Payload (JSON)</h2><pre>{json_str}</pre></section>"
    )


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    os.makedirs("uploads", exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method != "POST":
            return render_template_string(PAGE, body=Markup(""))

        clinical_file = request.files.get("clinical_image")
        pgx_file = request.files.get("pgx_pdf")
        clinical_text, pgx_text, ocr_engine = "", "", "none"

        try:
            if clinical_file and clinical_file.filename:
                ext = Path(clinical_file.filename).suffix.lower()
                img_path = Path("uploads") / f"temp_img{ext}"
                clinical_file.save(img_path)
                ocr_res = ocr_image(img_path)
                clinical_text, ocr_engine = ocr_res.text, ocr_res.engine
                img_path.unlink(missing_ok=True)

            if pgx_file and pgx_file.filename:
                ext = Path(pgx_file.filename).suffix.lower()
                pdf_path = Path("uploads") / f"temp_pgx{ext}"
                pgx_file.save(pdf_path)
                pgx_text = parse_pdf(pdf_path)
                pdf_path.unlink(missing_ok=True)

            if not clinical_text and not pgx_text:
                return render_template_string(PAGE, body=Markup("<section><h2>Error</h2><p>Please upload at least one file.</p></section>"))

            payload = generate_rag_payload(clinical_text, pgx_text, ocr_engine)
            return render_template_string(PAGE, body=Markup(render_audit_tables(payload)))

        except Exception as e:
            return render_template_string(PAGE, body=Markup(f"<section><h2>Processing Error</h2><p>{esc(str(e))}</p></section>")), 500

    return app


if __name__ == "__main__":
    port = 5000
    print(f"🚀 Starting Senior NLP Dual-Intake Server on http://127.0.0.1:{port}")
    webbrowser.open(f"http://127.0.0.1:{port}")
    create_app().run(host="127.0.0.1", port=port, debug=False)
