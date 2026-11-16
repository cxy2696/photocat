"""Physics-informed synthetic photocatalyst library.

No public DFT database is bundled with this repository, so the demo runs on a
generated library whose *structure* mirrors Materials Project / Open Catalyst
records: composition, crystal family, synthesis conditions, bandgap, band
positions, charge-separation efficiency, stability and hydrogen evolution rate
(HER).  Targets are produced by a transparent latent model with literature-like
trends and calibrated noise; known photocatalysts are kept as a separate
"real-data adapters in :mod:`photocat.data.external` cover public benchmarks.

This is a benchmark for the *algorithmic pipeline*, not a replacement for DFT
or experiment.  The loader in :mod:`photocat.data.io` accepts real CSV exports
with the same schema, so the models can be trained on Materials Project /
Open Catalyst dumps unchanged.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .elements import (
    ANION_EG_OFFSET,
    BY_SYMBOL,
    COCATALYST_FACTOR,
    COCATALYSTS,
    DOPANT_EFFECTS,
    DOPANT_LIBRARY,
    METHODS,
)

# ---------------------------------------------------------------------------
# Crystal families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    crystal_system: str
    base_eg: float
    base_stability: float
    base_cs: float
    structural_distortion: float
    weight: float


FAMILIES: dict[str, Family] = {
    "rutile": Family("rutile", "rutile MO2", "tetragonal", 3.00, 1.20, 0.55, 0.08, 0.16),
    "corundum": Family("corundum", "corundum M2O3", "rhombohedral", 2.40, 0.80, 0.45, 0.10, 0.10),
    "perovskite": Family("perovskite", "perovskite ABO3", "orthorhombic", 3.00, 0.90, 0.60, 0.06, 0.18),
    "scheelite": Family("scheelite", "scheelite ABO4", "tetragonal", 2.20, 1.00, 0.55, 0.12, 0.08),
    "spinel": Family("spinel", "spinel AB2O4", "cubic", 1.90, 1.00, 0.50, 0.14, 0.10),
    "carbon_nitride": Family(
        "carbon_nitride", "carbon nitride", "orthorhombic", 2.55, 1.30, 0.65, 0.05, 0.08
    ),
    "aurivillius": Family(
        "aurivillius", "aurivillius Bi2BO6", "orthorhombic", 2.50, 1.10, 0.60, 0.15, 0.06
    ),
    "oxynitride": Family(
        "oxynitride", "oxynitride", "orthorhombic", 2.20, 0.50, 0.55, 0.11, 0.06
    ),
    "sulfide": Family("sulfide", "metal sulfide", "hexagonal", 2.20, -1.20, 0.50, 0.10, 0.08),
    "oxyhalide": Family("oxyhalide", "bismuth oxyhalide", "tetragonal", 2.80, 0.90, 0.55, 0.09, 0.10),
    # Used by external datasets that carry composition but no crystal family.
    "unknown": Family("unknown", "unclassified composition", "unknown", 2.50, 0.80, 0.50, 0.10, 0.00),
}

HOLDOUT_FAMILIES: tuple[str, ...] = ("oxynitride", "sulfide")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Material:
    """One photocatalyst record, mirroring a Materials Project-style row."""

    id: str
    formula: str
    family: str
    crystal_system: str
    elements: tuple[tuple[str, float, str], ...]
    dopant: str
    dopant_fraction: float
    cocatalyst: str
    method: str
    ph: float
    surface_area: float
    strain: float
    defect: float
    heterojunction: str
    eg: float
    her: float
    stability: float
    cs_eff: float
    cb: float
    vb: float
    cocatalyst_loading: float = 0.0
    light_intensity: float = 0.0
    catalyst_loading: float = 0.0
    sacrificial_agent: str = "unknown"
    sacrificial_concentration: float = 0.0
    descriptors: dict[str, float] = field(default_factory=dict)
    source: str = "synthetic"
    reference: str = ""
    split: str = ""

    @property
    def label(self) -> str:
        return f"{self.formula}:{self.dopant}" if self.dopant != "none" else self.formula

    @property
    def element_symbols(self) -> tuple[str, ...]:
        return tuple(symbol for symbol, _, _ in self.elements)

    @property
    def cation_symbols(self) -> tuple[str, ...]:
        return tuple(
            symbol for symbol, _, role in self.elements if role in ("cation", "framework")
        )

    @property
    def anion_symbols(self) -> tuple[str, ...]:
        return tuple(symbol for symbol, _, role in self.elements if role == "anion")

    @property
    def sites(self) -> dict[str, float]:
        return {symbol: stoich for symbol, stoich, _ in self.elements}

    def to_row(self) -> dict[str, str]:
        row = {
            "id": self.id,
            "formula": self.formula,
            "label": self.label,
            "family": self.family,
            "crystal_system": self.crystal_system,
            "elements": ";".join(f"{s}:{q:g}:{r}" for s, q, r in self.elements),
            "dopant": self.dopant,
            "dopant_fraction": f"{self.dopant_fraction:.6g}",
            "cocatalyst": self.cocatalyst,
            "method": self.method,
            "ph": f"{self.ph:.1f}",
            "surface_area": f"{self.surface_area:.2f}",
            "strain": f"{self.strain:.6f}",
            "defect": f"{self.defect:.6f}",
            "heterojunction": self.heterojunction,
            "cocatalyst_loading": f"{self.cocatalyst_loading:.6g}",
            "light_intensity": f"{self.light_intensity:.6g}",
            "catalyst_loading": f"{self.catalyst_loading:.6g}",
            "sacrificial_agent": self.sacrificial_agent,
            "sacrificial_concentration": f"{self.sacrificial_concentration:.6g}",
            "eg": f"{self.eg:.4f}",
            "her": f"{self.her:.4f}",
            "stability": f"{self.stability:.4f}",
            "cs_eff": f"{self.cs_eff:.4f}",
            "cb": f"{self.cb:.4f}",
            "vb": f"{self.vb:.4f}",
            "descriptors": json.dumps(self.descriptors, sort_keys=True),
            "source": self.source,
            "reference": self.reference,
            "split": self.split,
        }
        return row

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Material":
        elements = tuple(
            (symbol, float(stoich), role)
            for symbol, stoich, role in (item.split(":") for item in row["elements"].split(";"))
        )
        return cls(
            id=row["id"],
            formula=row["formula"],
            family=row["family"],
            crystal_system=row["crystal_system"],
            elements=elements,
            dopant=row["dopant"],
            dopant_fraction=float(row["dopant_fraction"]),
            cocatalyst=row["cocatalyst"],
            method=row["method"],
            ph=float(row["ph"]),
            surface_area=float(row["surface_area"]),
            strain=float(row["strain"]),
            defect=float(row["defect"]),
            heterojunction=row["heterojunction"],
            eg=float(row["eg"]),
            her=float(row["her"]),
            stability=float(row["stability"]),
            cs_eff=float(row["cs_eff"]),
            cb=float(row["cb"]),
            vb=float(row["vb"]),
            cocatalyst_loading=float(row.get("cocatalyst_loading", 0.0)),
            light_intensity=float(row.get("light_intensity", 0.0)),
            catalyst_loading=float(row.get("catalyst_loading", 0.0)),
            sacrificial_agent=row.get("sacrificial_agent", "unknown"),
            sacrificial_concentration=float(row.get("sacrificial_concentration", 0.0)),
            descriptors=json.loads(row["descriptors"]),
            source=row.get("source", "synthetic"),
            reference=row.get("reference", ""),
            split=row.get("split", ""),
        )


# ---------------------------------------------------------------------------
# Composition sampling
# ---------------------------------------------------------------------------

_PEROVSKITE_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("Ca", "Sr", "Ba"), ("Ti", "Zr", "Sn")),
    (("La", "Bi"), ("Fe", "Co", "Mn", "Cr")),
    (("Na", "K"), ("Nb", "Ta")),
)

_SCHEELITE_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("Bi", "La", "Ce", "In"), ("V", "Nb", "Ta")),
)

_SPINEL_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("Zn", "Ni", "Cu", "Co", "Mg", "Fe"), ("Fe", "Al", "Cr", "Mn", "Co")),
)


def _weighted_choice(rng: np.random.Generator, items: Sequence[str], weights: Sequence[float]) -> str:
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    return str(rng.choice(np.asarray(items), p=probabilities))


def _format_site(pairs: Sequence[tuple[str, float]]) -> str:
    """Format a possibly mixed crystallographic site, e.g. ``Ca0.5Sr0.5``."""
    return "".join(
        f"{symbol}{'' if abs(fraction - 1.0) < 1e-9 else f'{fraction:g}'}"
        for symbol, fraction in pairs
    )


def _mix_site(
    rng: np.random.Generator, options: Sequence[str], mix_probability: float = 0.45
) -> list[tuple[str, float]]:
    """Pick one cation, or an isovalent two-cation solid solution."""
    if len(options) >= 2 and rng.random() < mix_probability:
        first, second = rng.choice(np.asarray(options), size=2, replace=False)
        fraction = float(rng.choice((0.25, 0.30, 0.50, 0.70, 0.75)))
        return [(str(first), fraction), (str(second), 1.0 - fraction)]
    return [(str(rng.choice(np.asarray(options))), 1.0)]


def _sample_composition(family: str, rng: np.random.Generator) -> tuple[str, list[tuple[str, float, str]]]:
    """Return (formula, [(symbol, stoichiometry, role), ...]).

    Isovalent site mixing creates solid solutions such as Ca0.5Sr0.5TiO3 and
    Ti0.7Sn0.3O2, which greatly expands the composition space without breaking
    charge neutrality.
    """
    if family == "rutile":
        site = _mix_site(rng, ("Ti", "Sn", "Zr", "Ce"))
        return (
            f"{_format_site(site)}O2",
            [(symbol, fraction, "cation") for symbol, fraction in site] + [("O", 2.0, "anion")],
        )

    if family == "corundum":
        site = _mix_site(rng, ("Fe", "Bi", "Cr", "In", "Ga", "Al", "V"))
        site_formula = _format_site(site)
        if len(site) > 1:
            site_formula = f"({site_formula})"
        return (
            f"{site_formula}2O3",
            [(symbol, 2.0 * fraction, "cation") for symbol, fraction in site]
            + [("O", 3.0, "anion")],
        )

    if family == "perovskite":
        a_options, b_options = _PEROVSKITE_PAIRS[rng.integers(len(_PEROVSKITE_PAIRS))]
        a_site = _mix_site(rng, a_options)
        b_site = _mix_site(rng, b_options)
        return (
            f"{_format_site(a_site)}{_format_site(b_site)}O3",
            [(symbol, fraction, "cation") for symbol, fraction in a_site]
            + [(symbol, fraction, "cation") for symbol, fraction in b_site]
            + [("O", 3.0, "anion")],
        )

    if family == "scheelite":
        a_options, b_options = _SCHEELITE_PAIRS[rng.integers(len(_SCHEELITE_PAIRS))]
        a_site = _mix_site(rng, a_options)
        b_site = _mix_site(rng, b_options)
        return (
            f"{_format_site(a_site)}{_format_site(b_site)}O4",
            [(symbol, fraction, "cation") for symbol, fraction in a_site]
            + [(symbol, fraction, "cation") for symbol, fraction in b_site]
            + [("O", 4.0, "anion")],
        )

    if family == "spinel":
        a_options, b_options = _SPINEL_PAIRS[rng.integers(len(_SPINEL_PAIRS))]
        a_site = _mix_site(rng, a_options)
        b_site = _mix_site(rng, b_options)
        b_formula = _format_site(b_site)
        if len(b_site) > 1:
            b_formula = f"({b_formula})"
        return (
            f"{_format_site(a_site)}{b_formula}2O4",
            [(symbol, fraction, "cation") for symbol, fraction in a_site]
            + [(symbol, 2.0 * fraction, "cation") for symbol, fraction in b_site]
            + [("O", 4.0, "anion")],
        )

    if family == "carbon_nitride":
        return "C3N4", [("C", 3.0, "framework"), ("N", 4.0, "anion")]

    if family == "aurivillius":
        b = str(rng.choice(("W", "Mo")))
        return f"Bi2{b}O6", [("Bi", 2.0, "cation"), (b, 1.0, "cation"), ("O", 6.0, "anion")]

    if family == "oxynitride":
        if rng.random() < 0.5:
            metal = str(rng.choice(("Ta", "Nb")))
            return f"{metal}ON", [(metal, 1.0, "cation"), ("O", 1.0, "anion"), ("N", 1.0, "anion")]
        a = str(rng.choice(("Ca", "Sr", "Ba", "La")))
        if a == "La":
            b_site = [("Ti", 1.0)]
        else:
            b_site = _mix_site(rng, ("Ta", "Nb"))
        return (
            f"{a}{_format_site(b_site)}O2N",
            [(a, 1.0, "cation")]
            + [(symbol, fraction, "cation") for symbol, fraction in b_site]
            + [("O", 2.0, "anion"), ("N", 1.0, "anion")],
        )

    if family == "sulfide":
        subtype = rng.integers(4)
        if subtype == 0:
            metal = str(rng.choice(("Cd", "Zn")))
            return f"{metal}S", [(metal, 1.0, "cation"), ("S", 1.0, "anion")]
        if subtype == 1:
            metal = str(rng.choice(("In", "Bi")))
            return f"{metal}2S3", [(metal, 2.0, "cation"), ("S", 3.0, "anion")]
        if subtype == 2:
            site = _mix_site(rng, ("In", "Ga"))
            return (
                f"Cu{_format_site(site)}S2",
                [("Cu", 1.0, "cation")]
                + [(symbol, fraction, "cation") for symbol, fraction in site]
                + [("S", 2.0, "anion")],
            )
        return "MoS2", [("Mo", 1.0, "cation"), ("S", 2.0, "anion")]

    if family == "oxyhalide":
        halide = str(rng.choice(("Cl", "Br", "I")))
        return f"BiO{halide}", [("Bi", 1.0, "cation"), ("O", 1.0, "anion"), (halide, 1.0, "anion")]

    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# Descriptor and target model
# ---------------------------------------------------------------------------


def _weighted_average(
    elements: Iterable[tuple[str, float, str]], attr: str, roles: tuple[str, ...]
) -> float:
    total = 0.0
    weight = 0.0
    for symbol, stoich, role in elements:
        if role not in roles:
            continue
        total += stoich * float(getattr(BY_SYMBOL[symbol], attr))
        weight += stoich
    return total / weight if weight > 0 else 0.0


def _weighted_moment(
    elements: Iterable[tuple[str, float, str]], attr: str, roles: tuple[str, ...]
) -> tuple[float, float]:
    values: list[tuple[float, float]] = []
    for symbol, stoich, role in elements:
        if role in roles:
            values.append((stoich, float(getattr(BY_SYMBOL[symbol], attr))))
    if not values:
        return 0.0, 0.0
    weight = sum(stoich for stoich, _ in values)
    mean = sum(stoich * value for stoich, value in values) / weight
    var = sum(stoich * (value - mean) ** 2 for stoich, value in values) / weight
    return mean, math.sqrt(max(var, 0.0))


def compute_descriptors(
    elements: Sequence[tuple[str, float, str]],
    family: str,
    ph: float,
    surface_area: float,
    strain: float,
    defect: float,
) -> dict[str, float]:
    """Composition and condition descriptors shared by data and features."""
    cat_roles = ("cation", "framework")
    chi_cat = _weighted_average(elements, "chi", cat_roles)
    radius_cat, radius_std = _weighted_moment(elements, "radius", cat_roles)
    d_cat = _weighted_average(elements, "d_electrons", cat_roles)
    valence_cat = _weighted_average(elements, "valence", cat_roles)
    chi_anion = _weighted_average(elements, "chi", ("anion",))
    abundance_min = min(BY_SYMBOL[symbol].abundance for symbol, _, _ in elements)
    n_elements = len({symbol for symbol, _, _ in elements})

    # Geometric distortion: Goldschmidt tolerance factor for perovskites,
    # radius mismatch elsewhere, plus a family-specific structural floor.
    distortion = FAMILIES[family].structural_distortion
    if family == "perovskite":
        a_pairs = [(q, BY_SYMBOL[s].radius) for s, q, r in elements if r == "cation" and BY_SYMBOL[s].radius > 0.9]
        b_pairs = [(q, BY_SYMBOL[s].radius) for s, q, r in elements if r == "cation" and BY_SYMBOL[s].radius <= 0.9]
        if a_pairs and b_pairs:
            a_weight = sum(q for q, _ in a_pairs)
            b_weight = sum(q for q, _ in b_pairs)
            r_a = sum(q * radius for q, radius in a_pairs) / a_weight
            r_b = sum(q * radius for q, radius in b_pairs) / b_weight
            r_o = BY_SYMBOL["O"].radius
            tolerance = (r_a + r_o) / (math.sqrt(2.0) * (r_b + r_o) + 1e-9)
            distortion += abs(tolerance - 1.0)
    distortion += radius_std / (radius_cat + 1e-6)

    return {
        "chi_cation": chi_cat,
        "radius_cation": radius_cat,
        "radius_std": radius_std,
        "d_cation": d_cat,
        "log_abundance_min": math.log10(abundance_min + 1.0),
        "chi_anion": chi_anion,
        "n_elements": float(n_elements),
        "distortion": float(min(distortion, 0.8)),
        "valence_cation": valence_cat,
        "ph": ph,
        "log_surface_area": math.log10(surface_area + 1.0),
        "strain": strain,
        "defect": defect,
    }


def _heterojunction_boost(kind: str) -> float:
    return {"none": 0.0, "type-II": 0.16, "Z-scheme": 0.10}[kind]


def compute_targets(
    elements: Sequence[tuple[str, float, str]],
    family: str,
    descriptors: dict[str, float],
    dopant: str,
    dopant_fraction: float,
    cocatalyst: str,
    ph: float,
    surface_area: float,
    strain: float,
    defect: float,
    heterojunction: str,
    rng: np.random.Generator,
    eg_noise: float,
    her_noise: float,
) -> tuple[float, float, float, float, float, float]:
    """Latent physical model: returns eg, her, stability, cs_eff, cb, vb."""
    spec = FAMILIES[family]
    dopant_eg, dopant_cs, dopant_stability = DOPANT_EFFECTS.get(dopant, (0.0, 0.0, 0.0))
    dopant_eg *= min(dopant_fraction / 0.05, 1.5)
    dopant_cs *= min(dopant_fraction / 0.05, 1.5)
    dopant_stability *= min(dopant_fraction / 0.05, 1.5)

    cat_roles = ("cation", "framework")
    cation_pairs = [
        (stoich, BY_SYMBOL[symbol].chi)
        for symbol, stoich, role in elements
        if role in cat_roles
    ]
    cation_weight = sum(weight for weight, _ in cation_pairs) or 1.0
    chi_values = [chi for _, chi in cation_pairs]
    bowing = 0.0
    if len(set(round(chi, 3) for chi in chi_values)) > 1:
        mean_chi = sum(w * c for w, c in cation_pairs) / cation_weight
        variance = sum(w * (c - mean_chi) ** 2 for w, c in cation_pairs) / cation_weight
        bowing = 0.55 * math.sqrt(variance)

    eg_offset = _weighted_average(elements, "eg_offset", cat_roles)
    anion_offset = sum(
        stoich * ANION_EG_OFFSET[BY_SYMBOL[symbol].anion_type]
        for symbol, stoich, role in elements
        if role == "anion"
    ) / max(sum(stoich for _, stoich, role in elements if role == "anion"), 1e-9)

    eg = (
        spec.base_eg
        + 0.45 * (1.75 - descriptors["chi_cation"])
        + eg_offset
        + anion_offset
        + 0.90 * descriptors["distortion"]
        - 0.60 * strain * 50.0
        - 0.55 * defect
        + dopant_eg
        - bowing
        + rng.normal(0.0, eg_noise)
    )
    eg = float(np.clip(eg, 0.55, 5.2))

    cb = (
        -0.10
        + 0.22 * (descriptors["chi_cation"] - 1.75)
        - 0.35 * descriptors["distortion"]
        + 0.015 * descriptors["d_cation"]
        + 0.10 * dopant_eg
    )
    vb = cb + eg

    drive_h = 1.0 / (1.0 + math.exp(-4.0 * (-0.05 - cb)))
    drive_o = 1.0 / (1.0 + math.exp(-4.0 * (vb - 1.23)))
    light = math.exp(-0.5 * ((eg - 1.95) / 0.62) ** 2)
    cs_struct = (
        spec.base_cs
        + 1.40 * (descriptors["distortion"] - spec.structural_distortion)
        + dopant_cs
        + _heterojunction_boost(heterojunction)
        - 1.20 * defect
    )
    cs_eff = 1.0 / (1.0 + math.exp(-4.0 * (cs_struct - 0.35)))
    surface_factor = (max(surface_area, 0.5) / 25.0) ** 0.30
    ph_factor = math.exp(-0.5 * ((ph - 7.0) / 3.2) ** 2)
    cocat_factor = COCATALYST_FACTOR[cocatalyst]
    her = (
        22.0
        * drive_h
        * drive_o
        * light
        * cs_eff
        * surface_factor
        * cocat_factor
        * ph_factor
        * math.exp(rng.normal(0.0, her_noise))
    )
    her = float(np.clip(her, 0.01, 80.0))

    stability_logit = (
        spec.base_stability
        + 0.80 * (1.0 - abs(ph - 7.0) / 6.0)
        - 1.50 * defect
        + 0.60 * dopant_stability
        - 0.50 * abs(strain) * 100.0
        + 0.35 * (1.0 - descriptors["distortion"])
        + rng.normal(0.0, 0.35)
    )
    stability = float(1.0 / (1.0 + math.exp(-stability_logit)))
    return eg, her, stability, cs_eff, float(cb), float(vb)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _sample_context(family: str, rng: np.random.Generator) -> dict[str, object]:
    if family == "carbon_nitride":
        method = "thermal-polymerization" if rng.random() < 0.8 else "hydrothermal"
        surface = float(np.clip(rng.lognormal(math.log(55.0), 0.55), 5.0, 400.0))
    elif family == "perovskite":
        method = _weighted_choice(
            rng, METHODS, (0.20, 0.30, 0.40, 0.05, 0.05)
        )
        surface = float(np.clip(rng.lognormal(math.log(12.0), 0.55), 1.0, 150.0))
    else:
        method = _weighted_choice(rng, METHODS, (0.45, 0.25, 0.15, 0.10, 0.05))
        surface = float(np.clip(rng.lognormal(math.log(25.0), 0.60), 1.0, 300.0))

    ph = float(np.clip(round(rng.normal(7.0, 2.4) * 2.0) / 2.0, 1.0, 13.0))
    strain = float(np.clip(rng.normal(0.0, 0.012), -0.05, 0.05))
    defect = float(np.clip(rng.beta(2.0, 5.0) * 0.55, 0.0, 0.45))
    heterojunction = _weighted_choice(rng, ("none", "type-II", "Z-scheme"), (0.80, 0.12, 0.08))
    cocatalyst = _weighted_choice(rng, COCATALYSTS, (0.25, 0.20, 0.20, 0.15, 0.12, 0.08))
    return {
        "method": method,
        "surface_area": surface,
        "ph": ph,
        "strain": strain,
        "defect": defect,
        "heterojunction": heterojunction,
        "cocatalyst": cocatalyst,
    }


def make_material(
    idx: int,
    family: str,
    elements: Sequence[tuple[str, float, str]],
    formula: str,
    context: dict[str, object],
    rng: np.random.Generator,
    eg_noise: float,
    her_noise: float,
    dopant: str = "none",
    dopant_fraction: float = 0.0,
    crystal_system: str | None = None,
    source: str = "synthetic",
) -> Material:
    elements = list(elements)
    if dopant != "none" and dopant_fraction > 0:
        elements.append((dopant, float(dopant_fraction), "dopant"))
    descriptors = compute_descriptors(
        elements,
        family,
        float(context["ph"]),
        float(context["surface_area"]),
        float(context["strain"]),
        float(context["defect"]),
    )
    eg, her, stability, cs_eff, cb, vb = compute_targets(
        elements,
        family,
        descriptors,
        dopant,
        dopant_fraction,
        str(context["cocatalyst"]),
        float(context["ph"]),
        float(context["surface_area"]),
        float(context["strain"]),
        float(context["defect"]),
        str(context["heterojunction"]),
        rng,
        eg_noise,
        her_noise,
    )
    return Material(
        id=f"pc-{idx:06d}",
        formula=formula,
        family=family,
        crystal_system=crystal_system or FAMILIES[family].crystal_system,
        elements=tuple(elements),
        dopant=dopant,
        dopant_fraction=float(dopant_fraction),
        cocatalyst=str(context["cocatalyst"]),
        method=str(context["method"]),
        ph=float(context["ph"]),
        surface_area=float(context["surface_area"]),
        strain=float(context["strain"]),
        defect=float(context["defect"]),
        heterojunction=str(context["heterojunction"]),
        eg=float(eg),
        her=float(her),
        stability=float(stability),
        cs_eff=float(cs_eff),
        cb=float(cb),
        vb=float(vb),
        descriptors=descriptors,
        source=source,
    )


def generate_materials(
    n: int = 12000,
    seed: int = 2025,
    eg_noise: float = 0.14,
    her_noise: float = 0.22,
    max_per_formula: int = 60,
) -> list[Material]:
    """Generate a de-duplicated library of synthetic photocatalysts."""
    rng = np.random.default_rng(seed)
    families = list(FAMILIES)
    weights = [FAMILIES[key].weight for key in families]
    probabilities = np.asarray(weights) / sum(weights)
    counts: dict[tuple[str, str], int] = {}
    materials: list[Material] = []

    attempts = 0
    while len(materials) < n and attempts < n * 40:
        attempts += 1
        family = str(rng.choice(np.asarray(families), p=probabilities))
        formula, elements = _sample_composition(family, rng)
        dopant = "none"
        dopant_fraction = 0.0
        if rng.random() < 0.45:
            dopant = str(rng.choice(np.asarray(DOPANT_LIBRARY)))
            dopant_fraction = float(rng.uniform(0.02, 0.10))
        key = (formula, dopant)
        if counts.get(key, 0) >= max_per_formula:
            continue
        counts[key] = counts.get(key, 0) + 1

        context = _sample_context(family, rng)
        crystal = FAMILIES[family].crystal_system
        if family == "perovskite" and rng.random() < 0.2:
            crystal = "cubic"
        material = make_material(
            len(materials) + 1,
            family,
            elements,
            formula,
            context,
            rng,
            eg_noise,
            her_noise,
            dopant=dopant,
            dopant_fraction=dopant_fraction,
            crystal_system=crystal,
        )
        materials.append(material)

    if len(materials) < n:
        raise RuntimeError(f"could only generate {len(materials)} / {n} unique materials")
    return materials


# ---------------------------------------------------------------------------
# Splitting and I/O
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    materials: list[Material]

    def split(self, name: str) -> list[Material]:
        return [material for material in self.materials if material.split == name]

    @property
    def train(self) -> list[Material]:
        return self.split("train")

    @property
    def val(self) -> list[Material]:
        return self.split("val")

    @property
    def test(self) -> list[Material]:
        return self.split("test")

    @property
    def ood(self) -> list[Material]:
        return self.split("ood")


def assign_splits(
    materials: Sequence[Material],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    holdout_families: Sequence[str] = HOLDOUT_FAMILIES,
    seed: int = 2025,
) -> None:
    """Group-aware split: all condition variants of a formula stay together."""
    holdout = set(holdout_families)
    rng = np.random.default_rng(seed + 11)
    by_family: dict[str, dict[tuple[str, str], list[Material]]] = {}
    for material in materials:
        if material.family in holdout:
            material.split = "ood"
            continue
        key = (material.formula, material.dopant)
        by_family.setdefault(material.family, {}).setdefault(key, []).append(material)

    # Stratify the group split by crystal family so every visible family is
    # represented in train, validation and test.
    for family in sorted(by_family):
        groups = by_family[family]
        keys = sorted(groups)
        rng.shuffle(keys)
        n_groups = len(keys)
        n_test = max(1, int(round(n_groups * test_frac))) if n_groups >= 3 else 0
        n_val = max(1, int(round(n_groups * val_frac))) if n_groups >= 3 else 0
        n_test = min(n_test, n_groups - 1)
        n_val = min(n_val, n_groups - n_test - 1)
        for i, key in enumerate(keys):
            if i < n_groups - n_test - n_val:
                split = "train"
            elif i < n_groups - n_test:
                split = "val"
            else:
                split = "test"
            for material in groups[key]:
                material.split = split


def generate_dataset(
    n: int = 12000,
    seed: int = 2025,
    eg_noise: float = 0.14,
    her_noise: float = 0.22,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    holdout_families: Sequence[str] = HOLDOUT_FAMILIES,
) -> DatasetBundle:
    materials = generate_materials(n=n, seed=seed, eg_noise=eg_noise, her_noise=her_noise)
    assign_splits(materials, val_frac=val_frac, test_frac=test_frac, holdout_families=holdout_families, seed=seed)
    return DatasetBundle(materials=materials)


def save_materials(path: str | Path, materials: Sequence[Material]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materials[0].to_row())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for material in materials:
            writer.writerow(material.to_row())


def load_materials(path: str | Path) -> list[Material]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [Material.from_row(row) for row in csv.DictReader(handle)]


def summary(materials: Sequence[Material]) -> dict[str, object]:
    eg = np.asarray([m.eg for m in materials])
    her = np.asarray([m.her for m in materials])
    stability = np.asarray([m.stability for m in materials])
    candidates = (eg >= 1.23) & (eg <= 3.0) & (her >= 6.0) & (stability >= 0.55)
    return {
        "n": len(materials),
        "families": {key: sum(m.family == key for m in materials) for key in FAMILIES},
        "eg_mean": float(eg.mean()),
        "eg_median": float(np.median(eg)),
        "her_mean": float(her.mean()),
        "her_median": float(np.median(her)),
        "stability_mean": float(stability.mean()),
        "candidate_rate": float(candidates.mean()),
    }
