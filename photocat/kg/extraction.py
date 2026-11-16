"""Rule-based NER and relation extraction for photocatalyst sentences.

This is a transparent stand-in for a SciBERT-based extractor: formulas, numeric
properties, pH, methods, dopants, co-catalysts and crystal systems are matched
with regular expressions and linked to material entities.  It is evaluated
against the gold annotations in :mod:`photocat.kg.corpus` so the extraction
quality is measured rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..data.elements import BY_SYMBOL
from ..data.synth import Material
from ..data.elements import COCATALYSTS, METHODS
from .store import KGStore
from .ontology import (
    Triple,
    cocatalyst_entity,
    crystal_entity,
    dopant_entity,
    material_entity,
    method_entity,
)

FORMULA_RE = re.compile(
    r"(?<![A-Za-z])(?:g-)?(?:[A-Z][a-z]?\d*\.?\d*|\([^()\s]{1,20}\)\d*\.?\d*){2,}(?::[A-Z][a-z]?)?"
)
BANDGAP_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:eV)\b")
HER_VALUE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mmol\s*g\s*[-−–]?\s*1\s*h\s*[-−–]?\s*1"
)
PH_RE = re.compile(r"pH\s*(?:of|=|~|at)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
DOPANT_RE = re.compile(r"doped\s+with\s+([A-Z][a-z]?)")
COCATALYST_PHRASE = re.compile(r"co-?catalyst", re.IGNORECASE)
CRYSTAL_TERMS = ("tetragonal", "rhombohedral", "orthorhombic", "cubic", "hexagonal")


@dataclass
class ExtractionResult:
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    entities: list[tuple[str, str]] = field(default_factory=list)
    subject: str | None = None


def _valid_formula(candidate: str) -> bool:
    base = candidate
    if ":" in base:
        base, dopant = base.split(":", 1)
        if dopant not in BY_SYMBOL:
            return False
    symbols = re.findall(r"[A-Z][a-z]?", base)
    distinct = set(symbols)
    if len(distinct) < 2:
        return False
    return all(symbol in BY_SYMBOL for symbol in distinct)


def _formula_mentions(text: str) -> list[tuple[int, str]]:
    mentions: list[tuple[int, str]] = []
    for match in FORMULA_RE.finditer(text):
        candidate = match.group(0).strip()
        if not _valid_formula(candidate):
            continue
        # Skip co-catalyst formula mentions such as "MoS2 as a co-catalyst".
        suffix = text[match.end() : match.end() + 30].lower()
        if "co-catalyst" in suffix or "cocatalyst" in suffix:
            continue
        mentions.append((match.start(), candidate))
    return mentions


def extract(text: str) -> ExtractionResult:
    result = ExtractionResult()
    mentions = _formula_mentions(text)
    subject = mentions[-1][1] if mentions else None
    result.subject = subject

    bandgap = BANDGAP_VALUE_RE.search(text)
    her = HER_VALUE_RE.search(text)
    ph = PH_RE.search(text)
    dopant = DOPANT_RE.search(text)
    method = next((m for m in METHODS if m.lower() in text.lower()), None)
    crystal = next((c for c in CRYSTAL_TERMS if c in text.lower()), None)
    cocatalyst = None
    if COCATALYST_PHRASE.search(text):
        for candidate in COCATALYSTS:
            if candidate == "none":
                continue
            # Word-ish match; MoS2/NiS/Pt/Co-Pi/Ni2P are distinctive strings.
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])", text):
                cocatalyst = candidate
                break

    entities: list[tuple[str, str]] = []
    for _, mention in mentions:
        entities.append((mention, "Material"))
    if bandgap:
        entities.append((bandgap.group(1), "Bandgap"))
    if her:
        entities.append((her.group(1), "HER"))
    if ph:
        entities.append((ph.group(1), "PH"))
    if method:
        entities.append((method, "SynthesisMethod"))
    if dopant and dopant.group(1) in BY_SYMBOL:
        entities.append((dopant.group(1), "Dopant"))
    if crystal:
        entities.append((crystal, "CrystalSystem"))
    if cocatalyst:
        entities.append((cocatalyst, "CoCatalyst"))
    result.entities = entities

    if subject is not None:
        if bandgap:
            result.triples.append((subject, "hasBandgap", f"{float(bandgap.group(1)):.2f}"))
        if her:
            result.triples.append((subject, "hasHER", f"{float(her.group(1)):.1f}"))
        if ph:
            result.triples.append((subject, "operatesAtPH", f"{float(ph.group(1)):.1f}"))
        if method:
            result.triples.append((subject, "hasSynthesisMethod", method))
        if dopant and dopant.group(1) in BY_SYMBOL:
            result.triples.append((subject, "hasDopant", dopant.group(1)))
        if crystal:
            result.triples.append((subject, "hasCrystalSystem", crystal))
        if cocatalyst:
            result.triples.append((subject, "hasCoCatalyst", cocatalyst))
    return result


def evaluate_extraction(
    records: Sequence[object], predictions: Sequence[ExtractionResult]
) -> dict[str, float]:
    """Micro precision/recall/F1 over relation triples and entity mentions."""
    true_positive = false_positive = false_negative = 0
    entity_tp = entity_fp = entity_fn = 0
    for record, prediction in zip(records, predictions):
        gold = set(tuple(triple) for triple in record.gold)  # type: ignore[attr-defined]
        predicted = set(prediction.triples)
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)

        gold_entities = {
            (triple[0], "Material") for triple in gold
        } | {(triple[2], _entity_type(triple[1])) for triple in gold}
        predicted_entities = set(prediction.entities)
        entity_tp += len(gold_entities & predicted_entities)
        entity_fp += len(predicted_entities - gold_entities)
        entity_fn += len(gold_entities - predicted_entities)

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }

    return {
        "relations": prf(true_positive, false_positive, false_negative),
        "entities": prf(entity_tp, entity_fp, entity_fn),
        "n_sentences": float(len(predictions)),
    }


def _entity_type(relation: str) -> str:
    return {
        "hasBandgap": "Bandgap",
        "hasHER": "HER",
        "operatesAtPH": "PH",
        "hasSynthesisMethod": "SynthesisMethod",
        "hasDopant": "Dopant",
        "hasCrystalSystem": "CrystalSystem",
        "hasCoCatalyst": "CoCatalyst",
    }.get(relation, "Value")


def load_extractions(
    store: KGStore,
    records: Sequence[object],
    material_by_label: dict[str, Material],
    source: str = "literature_extraction",
) -> dict[str, float]:
    """Add extracted triples to the KG and report the entity-link rate."""
    added = 0
    linked = 0
    total = 0
    for record in records:
        text = record.text  # type: ignore[attr-defined]
        result = extract(text)
        material = material_by_label.get(result.subject or "")
        total += len(result.triples)
        if material is None:
            continue
        subject = material_entity(material)
        linked += len(result.triples)
        for _, relation, obj in result.triples:
            if relation == "hasBandgap":
                triple = Triple(subject, relation, obj, value=float(obj), unit="eV", source=source)
            elif relation == "hasHER":
                triple = Triple(
                    subject, relation, obj, value=float(obj), unit="mmol g-1 h-1", source=source
                )
            elif relation == "operatesAtPH":
                triple = Triple(subject, relation, obj, value=float(obj), unit="pH", source=source)
            elif relation == "hasSynthesisMethod":
                store.add_entities([(method_entity(obj), obj, "SynthesisMethod")])
                triple = Triple(subject, relation, method_entity(obj), source=source)
            elif relation == "hasDopant":
                store.add_entities([(dopant_entity(obj), obj, "Dopant")])
                triple = Triple(subject, relation, dopant_entity(obj), source=source)
            elif relation == "hasCrystalSystem":
                store.add_entities([(crystal_entity(obj), obj, "CrystalSystem")])
                triple = Triple(subject, relation, crystal_entity(obj), source=source)
            elif relation == "hasCoCatalyst":
                store.add_entities([(cocatalyst_entity(obj), obj, "CoCatalyst")])
                triple = Triple(subject, relation, cocatalyst_entity(obj), source=source)
            else:
                triple = Triple(subject, relation, obj, source=source)
            store.add_triples([triple])
            added += 1
    store.connection.commit()
    return {
        "sentences": float(len(records)),
        "triples_extracted": float(total),
        "triples_linked": float(added),
        "link_rate": float(linked / max(total, 1)),
    }
