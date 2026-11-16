"""A small synthetic literature corpus with gold annotations.

Sentences are generated from library records using several phrasings that cover
the relation types the extractor must recover.  The corpus is explicitly
synthetic: it is a controlled benchmark for the NER/RE layer, not a set of real
publications.  Gold triples use the same normalisation as the extractor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.synth import FAMILIES, Material

CRYSTAL_SYSTEMS = ("tetragonal", "rhombohedral", "orthorhombic", "cubic", "hexagonal")


@dataclass(frozen=True)
class CorpusRecord:
    text: str
    material_id: str
    template: str
    gold: tuple[tuple[str, str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "material_id": self.material_id,
            "template": self.template,
            "gold": [list(triple) for triple in self.gold],
        }


def _template_record(material: Material, template: int) -> tuple[str, list[tuple[str, str, str]]] | None:
    label = material.label
    method = material.method
    ph = f"{material.ph:.1f}"
    eg = f"{material.eg:.2f}"
    her = f"{material.her:.1f}"
    gold: list[tuple[str, str, str]] = []

    if template == 1:
        text = (
            f"{label} exhibits a bandgap of {eg} eV and a hydrogen evolution rate of "
            f"{her} mmol g-1 h-1 under visible light."
        )
        gold = [(label, "hasBandgap", eg), (label, "hasHER", her)]
    elif template == 2:
        text = (
            f"{label} was prepared by a {method} method and shows a bandgap of {eg} eV "
            f"at pH {ph}."
        )
        gold = [
            (label, "hasSynthesisMethod", method),
            (label, "hasBandgap", eg),
            (label, "operatesAtPH", ph),
        ]
    elif template == 3:
        if material.cocatalyst != "none":
            text = (
                f"With {material.cocatalyst} as a co-catalyst, {label} reaches a hydrogen "
                f"evolution rate of {her} mmol g-1 h-1 at pH {ph}."
            )
            gold = [
                (label, "hasCoCatalyst", material.cocatalyst),
                (label, "hasHER", her),
                (label, "operatesAtPH", ph),
            ]
        else:
            text = (
                f"Without a co-catalyst, {label} reaches a hydrogen evolution rate of {her} "
                f"mmol g-1 h-1 at pH {ph}."
            )
            gold = [(label, "hasHER", her), (label, "operatesAtPH", ph)]
    elif template == 4:
        if material.dopant == "none":
            return None
        text = (
            f"{label} is doped with {material.dopant} and crystallizes in the "
            f"{material.crystal_system} system, with a bandgap of {eg} eV."
        )
        gold = [
            (label, "hasDopant", material.dopant),
            (label, "hasCrystalSystem", material.crystal_system),
            (label, "hasBandgap", eg),
        ]
    elif template == 5:
        text = (
            f"At pH {ph}, {label} exhibits a bandgap of {eg} eV and a hydrogen evolution "
            f"rate of {her} mmol g-1 h-1."
        )
        gold = [
            (label, "operatesAtPH", ph),
            (label, "hasBandgap", eg),
            (label, "hasHER", her),
        ]
    elif template == 6:
        family_label = FAMILIES[material.family].label
        text = (
            f"The {family_label} photocatalyst {label} was synthesized by a {method} route "
            f"and shows a hydrogen evolution rate of {her} mmol g-1 h-1."
        )
        gold = [(label, "hasSynthesisMethod", method), (label, "hasHER", her)]
    else:
        raise ValueError(template)
    return text, gold


def generate_corpus(
    materials: Sequence[Material],
    anchors: Sequence[Material] = (),
    n: int = 500,
    seed: int = 2025,
) -> list[CorpusRecord]:
    rng = np.random.default_rng(seed)
    selected = list(materials)
    if len(selected) > n:
        index = rng.choice(len(selected), n, replace=False)
        selected = [selected[i] for i in sorted(index)]
    selected.extend(anchors)
    records: list[CorpusRecord] = []
    for material in selected:
        for _ in range(20):
            template = int(rng.integers(1, 7))
            rendered = _template_record(material, template)
            if rendered is not None:
                text, gold = rendered
                records.append(
                    CorpusRecord(
                        text=text,
                        material_id=material.id,
                        template=str(template),
                        gold=tuple(gold),
                    )
                )
                break
    rng.shuffle(records)
    return records


def save_corpus(path: str | Path, records: Sequence[CorpusRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(
                CorpusRecord(
                    text=payload["text"],
                    material_id=payload["material_id"],
                    template=payload["template"],
                    gold=tuple(tuple(triple) for triple in payload["gold"]),
                )
            )
    return records
