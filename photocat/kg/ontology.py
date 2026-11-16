"""Ontology and triple construction for the photocatalytic materials KG.

The graph keeps a Materials-Project-like structured layer (material, element,
crystal family, property values, synthesis conditions) and a literature layer
(entities and relations extracted from text).  Entity ids are namespaced
strings so that numeric values and categorical links can coexist in one
SQLite triple store.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.synth import FAMILIES, Material

ENTITY_TYPES: tuple[str, ...] = (
    "Material",
    "Element",
    "Family",
    "CrystalSystem",
    "CoCatalyst",
    "SynthesisMethod",
    "Dopant",
    "Application",
    "DesignTarget",
    "Dataset",
    "Paper",
)

RELATIONS: tuple[str, ...] = (
    "hasElement",
    "hasFamily",
    "hasCrystalSystem",
    "hasCoCatalyst",
    "hasSynthesisMethod",
    "hasDopant",
    "hasBandgap",
    "hasHER",
    "hasStability",
    "hasChargeSeparation",
    "hasConductionBand",
    "hasValenceBand",
    "operatesAtPH",
    "usedFor",
    "reportedIn",
    "hasLowerBound",
    "hasUpperBound",
    "hasMinimum",
    "hasUnit",
    "inferred:Candidate",
    "inferred:EarthAbundant",
    "inferred:Novel",
    "inferred:HeterojunctionWith",
)

APPLICATION_HYDROGEN = "application:hydrogen_evolution"
TARGET_BANDGAP = "target:bandgap_window"
TARGET_HER = "target:her_threshold"
DATASET_SYNTHETIC = "dataset:synthetic_library_v1"
DATASET_LITERATURE = "dataset:literature_seed"


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    value: float | None = None
    unit: str = ""
    source: str = "structured"
    confidence: float = 1.0

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


def material_entity(material: Material) -> str:
    return f"material:{material.id}"


def element_entity(symbol: str) -> str:
    return f"element:{symbol}"


def family_entity(key: str) -> str:
    return f"family:{key}"


def crystal_entity(name: str) -> str:
    return f"crystal_system:{name}"


def cocatalyst_entity(name: str) -> str:
    return f"cocatalyst:{name}"


def method_entity(name: str) -> str:
    return f"method:{name}"


def dopant_entity(symbol: str) -> str:
    return f"dopant:{symbol}"


def entity_definitions(material: Material) -> list[tuple[str, str, str]]:
    """(id, human name, type) for the material and its graph neighbours."""
    entities: list[tuple[str, str, str]] = [
        (material_entity(material), material.label, "Material"),
        (family_entity(material.family), FAMILIES[material.family].label, "Family"),
        (crystal_entity(material.crystal_system), material.crystal_system, "CrystalSystem"),
        (cocatalyst_entity(material.cocatalyst), material.cocatalyst, "CoCatalyst"),
        (method_entity(material.method), material.method, "SynthesisMethod"),
        (
            DATASET_LITERATURE if material.source == "literature_seed" else DATASET_SYNTHETIC,
            "Literature seed set" if material.source == "literature_seed" else "Synthetic library v1",
            "Dataset",
        ),
        (APPLICATION_HYDROGEN, "Photocatalytic hydrogen evolution", "Application"),
    ]
    for symbol, _, _ in material.elements:
        entities.append((element_entity(symbol), symbol, "Element"))
    if material.dopant != "none":
        entities.append((dopant_entity(material.dopant), material.dopant, "Dopant"))
    return entities


def material_triples(material: Material, source: str | None = None) -> list[Triple]:
    subject = material_entity(material)
    provenance = source or (
        "literature_seed" if material.source == "literature_seed" else "structured"
    )
    dataset = DATASET_LITERATURE if material.source == "literature_seed" else DATASET_SYNTHETIC
    total = sum(stoich for _, stoich, _ in material.elements) or 1.0
    triples: list[Triple] = [
        Triple(subject, "hasFamily", family_entity(material.family), source=provenance),
        Triple(
            subject,
            "hasCrystalSystem",
            crystal_entity(material.crystal_system),
            source=provenance,
        ),
        Triple(
            subject,
            "hasCoCatalyst",
            cocatalyst_entity(material.cocatalyst),
            source=provenance,
        ),
        Triple(
            subject,
            "hasSynthesisMethod",
            method_entity(material.method),
            source=provenance,
        ),
        Triple(
            subject,
            "hasBandgap",
            f"{material.eg:.3f}",
            value=material.eg,
            unit="eV",
            source=provenance,
        ),
        Triple(
            subject,
            "hasHER",
            f"{material.her:.3f}",
            value=material.her,
            unit="mmol g-1 h-1",
            source=provenance,
        ),
        Triple(
            subject,
            "hasStability",
            f"{material.stability:.3f}",
            value=material.stability,
            unit="probability",
            source=provenance,
        ),
        Triple(
            subject,
            "hasChargeSeparation",
            f"{material.cs_eff:.3f}",
            value=material.cs_eff,
            unit="probability",
            source=provenance,
        ),
        Triple(
            subject,
            "hasConductionBand",
            f"{material.cb:.3f}",
            value=material.cb,
            unit="eV vs NHE",
            source=provenance,
        ),
        Triple(
            subject,
            "hasValenceBand",
            f"{material.vb:.3f}",
            value=material.vb,
            unit="eV vs NHE",
            source=provenance,
        ),
        Triple(
            subject,
            "operatesAtPH",
            f"{material.ph:.1f}",
            value=material.ph,
            unit="pH",
            source=provenance,
        ),
        Triple(subject, "usedFor", APPLICATION_HYDROGEN, source=provenance),
        Triple(subject, "reportedIn", dataset, source=provenance),
    ]
    for symbol, stoich, _ in material.elements:
        triples.append(
            Triple(
                subject,
                "hasElement",
                element_entity(symbol),
                value=stoich / total,
                unit="fraction",
                source=provenance,
            )
        )
    if material.dopant != "none":
        triples.append(
            Triple(
                subject,
                "hasDopant",
                dopant_entity(material.dopant),
                value=material.dopant_fraction,
                unit="fraction",
                source=provenance,
            )
        )
    return triples


def design_target_triples(eg_window: tuple[float, float] = (1.23, 3.0), her_min: float = 6.0) -> list[Triple]:
    return [
        Triple(TARGET_BANDGAP, "hasLowerBound", f"{eg_window[0]:.2f}", value=eg_window[0], unit="eV"),
        Triple(TARGET_BANDGAP, "hasUpperBound", f"{eg_window[1]:.2f}", value=eg_window[1], unit="eV"),
        Triple(TARGET_BANDGAP, "hasUnit", "eV"),
        Triple(TARGET_HER, "hasMinimum", f"{her_min:.1f}", value=her_min, unit="mmol g-1 h-1"),
        Triple(TARGET_HER, "hasUnit", "mmol g-1 h-1"),
        Triple(TARGET_BANDGAP, "usedFor", APPLICATION_HYDROGEN),
        Triple(TARGET_HER, "usedFor", APPLICATION_HYDROGEN),
    ]


def design_target_entities() -> list[tuple[str, str, str]]:
    return [
        (TARGET_BANDGAP, "Photocatalytic bandgap window", "DesignTarget"),
        (TARGET_HER, "HER performance threshold", "DesignTarget"),
        (APPLICATION_HYDROGEN, "Photocatalytic hydrogen evolution", "Application"),
    ]
