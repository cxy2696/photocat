"""Rule-based semantic reasoning over the KG.

Rules materialise interpretable triples:

* candidate: 1.23 <= Eg <= 3.0 eV, HER >= 6, stability >= 0.55;
* earth-abundant: every constituent above a crustal-abundance threshold;
* novel: Tanimoto similarity to the training library < 0.8;
* type-II heterojunction: staggered band alignment with a shared element.

Every inferred edge carries a derivation so the ranking can be explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from ..data.elements import BY_SYMBOL
from ..ranking import RankConfig
from .ontology import (
    APPLICATION_HYDROGEN,
    Triple,
)
from .store import KGStore

EARTH_ABUNDANCE_PPM = 10.0


@dataclass
class ReasoningResult:
    candidates: list[dict[str, object]] = field(default_factory=list)
    earth_abundant: list[dict[str, object]] = field(default_factory=list)
    novel: list[dict[str, object]] = field(default_factory=list)
    heterojunctions: list[dict[str, object]] = field(default_factory=list)
    inferred: list[Triple] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def _element_map(store: KGStore) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for row in store.connection.execute(
        "SELECT subject, object FROM triples WHERE relation = 'hasElement'"
    ):
        material_id = str(row["subject"]).split(":", 1)[1]
        symbol = str(row["object"]).split(":", 1)[1]
        mapping.setdefault(material_id, set()).add(symbol)
    return mapping


def is_candidate(row: dict[str, object], config: RankConfig) -> bool:
    return (
        config.eg_min <= float(row["eg"]) <= config.eg_max
        and float(row["her"]) >= config.her_min
        and float(row["stability"]) >= config.stability_min
    )


def is_earth_abundant(elements: set[str], threshold: float = EARTH_ABUNDANCE_PPM) -> bool:
    return bool(elements) and all(
        BY_SYMBOL[symbol].abundance >= threshold for symbol in elements if symbol in BY_SYMBOL
    )


def staggered(row_a: dict[str, object], row_b: dict[str, object]) -> bool:
    """Type-II (staggered) alignment: CBM of one lies between the other's CBM/VBM."""
    cb_a, vb_a = float(row_a["cb"]), float(row_a["vb"])
    cb_b, vb_b = float(row_b["cb"]), float(row_b["vb"])
    return (cb_a < cb_b < vb_a < vb_b) or (cb_b < cb_a < vb_b < vb_a)


def run_reasoning(
    store: KGStore,
    config: RankConfig,
    similarity: dict[str, float] | None = None,
    max_heterojunctions: int = 30,
    materialise: bool = True,
) -> ReasoningResult:
    elements = _element_map(store)
    rows = store.query_materials(
        eg=(config.eg_min, config.eg_max),
        her_min=config.her_min,
        stability_min=config.stability_min,
    )
    result = ReasoningResult()
    result.candidates = rows

    for row in rows:
        material_id = str(row["material_id"])
        if is_earth_abundant(elements.get(material_id, set())):
            result.earth_abundant.append(row)
        if similarity is not None and similarity.get(material_id, 1.0) < config.novelty_max:
            result.novel.append(row)

    # Type-II heterojunctions among the highest-HER candidates sharing an element.
    pool = rows[:400]
    pair_count = 0
    seen: set[tuple[str, str]] = set()
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if pair_count >= max_heterojunctions:
                break
            a, b = pool[i], pool[j]
            id_a, id_b = str(a["material_id"]), str(b["material_id"])
            if id_a == id_b:
                continue
            if not (elements.get(id_a, set()) & elements.get(id_b, set())):
                continue
            if not staggered(a, b):
                continue
            key = tuple(sorted((id_a, id_b)))
            if key in seen:
                continue
            seen.add(key)
            joint = min(float(a["her"]), float(b["her"]))
            result.heterojunctions.append(
                {
                    "a": id_a,
                    "b": id_b,
                    "label_a": a["label"],
                    "label_b": b["label"],
                    "eg_a": float(a["eg"]),
                    "eg_b": float(b["eg"]),
                    "her_a": float(a["her"]),
                    "her_b": float(b["her"]),
                    "joint_her": joint,
                    "shared_elements": sorted(elements[id_a] & elements[id_b]),
                }
            )
            pair_count += 1
        if pair_count >= max_heterojunctions:
            break

    if materialise:
        store.remove_source("rule:candidate")
        store.remove_source("rule:earth_abundant")
        store.remove_source("rule:novel")
        store.remove_source("rule:heterojunction")
        triples: list[Triple] = []
        for row in result.candidates:
            subject = f"material:{row['material_id']}"
            triples.append(Triple(subject, "inferred:Candidate", "true", source="rule:candidate"))
        for row in result.earth_abundant:
            subject = f"material:{row['material_id']}"
            triples.append(
                Triple(subject, "inferred:EarthAbundant", "true", source="rule:earth_abundant")
            )
        for row in result.novel:
            subject = f"material:{row['material_id']}"
            triples.append(Triple(subject, "inferred:Novel", "true", source="rule:novel"))
        for pair in result.heterojunctions:
            subject_a = f"material:{pair['a']}"
            subject_b = f"material:{pair['b']}"
            triples.append(
                Triple(
                    subject_a,
                    "inferred:HeterojunctionWith",
                    subject_b,
                    source="rule:heterojunction",
                )
            )
            triples.append(
                Triple(
                    subject_b,
                    "inferred:HeterojunctionWith",
                    subject_a,
                    source="rule:heterojunction",
                )
            )
        store.add_triples(triples)
        store.connection.commit()
        seen: set[tuple[str, str, str]] = set()
        for triple in triples:
            key = triple.key()
            if key not in seen:
                seen.add(key)
                result.inferred.append(triple)

    result.stats = {
        "candidates": float(len(result.candidates)),
        "earth_abundant": float(len(result.earth_abundant)),
        "novel": float(len(result.novel)),
        "heterojunctions": float(len(result.heterojunctions)),
        "inferred_triples": float(len(result.inferred)),
    }
    return result


def explain(store: KGStore, material_id: str, max_partners: int = 3) -> dict[str, object]:
    """Human-readable explanation with evidence triples and reasoning paths."""
    row = store.material(material_id)
    if row is None:
        raise KeyError(f"unknown material {material_id!r}")
    subject = f"material:{material_id}"
    triples = store.triples(subject=subject)
    inferred = [triple for triple in triples if str(triple["relation"]).startswith("inferred:")]
    direct = [triple for triple in triples if not str(triple["relation"]).startswith("inferred:")]
    partners = [
        str(triple["object"]).split(":", 1)[1]
        for triple in inferred
        if triple["relation"] == "inferred:HeterojunctionWith"
    ][:max_partners]
    paths = store.find_paths(subject, APPLICATION_HYDROGEN, max_depth=2, max_paths=5)
    return {
        "material": row,
        "direct_triples": direct,
        "inferred_triples": inferred,
        "heterojunction_partners": partners,
        "paths_to_application": paths,
        "summary": (
            f"{row['label']} ({row['family']}) has Eg = {float(row['eg']):.2f} eV, "
            f"HER = {float(row['her']):.2f} mmol g-1 h-1 and stability "
            f"{float(row['stability']):.2f}; CB/VB = {float(row['cb']):.2f}/"
            f"{float(row['vb']):.2f} eV vs NHE at pH {float(row['ph']):.1f}."
        ),
    }


def format_explanation(explanation: dict[str, object]) -> str:
    row = explanation["material"]
    lines = [f"### {row['label']}", "", str(explanation["summary"]), ""]
    inferred = explanation["inferred_triples"]  # type: ignore[assignment]
    groups: dict[str, list[dict[str, object]]] = {}
    for triple in inferred:  # type: ignore[union-attr]
        groups.setdefault(str(triple["relation"]), []).append(triple)
    for relation, triples in groups.items():
        if relation == "inferred:HeterojunctionWith":
            partners = ", ".join(str(item["object"]).split(":", 1)[1] for item in triples[:4])
            lines.append(f"- inferred **{relation}** × {len(triples)} ({partners} …)")
        else:
            lines.append(f"- inferred **{relation}** ({triples[0]['source']})")
    lines.append("")
    lines.append("Evidence:")
    for triple in explanation["direct_triples"]:  # type: ignore[union-attr]
        unit = f" {triple['unit']}" if triple["unit"] else ""
        value = f" = {triple['value']}{unit}" if triple["value"] is not None else ""
        lines.append(f"- `{triple['relation']}` → {triple['object']}{value}")
    if explanation["heterojunction_partners"]:
        lines.append("")
        lines.append("Type-II partners: " + ", ".join(explanation["heterojunction_partners"]))
    return "\n".join(lines)
