"""Adapters for real, citable materials datasets.

Two datasets are supported:

``matbench_expt_gap``
    4,604 experimental band gaps for inorganic compositions, distributed by
    Matbench under the MIT licence (Figshare DOI 10.6084/m9.figshare.9765779;
    derived from Zhuo et al., J. Phys. Chem. Lett. 2018).

``ees_her``
    920 photocatalytic hydrogen-evolution records over TiO2-based materials,
    compiled from 52 papers in the EES Catalysis 2024 study by Haghshenas et
    al.  The upstream repository does not declare a redistribution licence, so
    the workbook is downloaded locally by ``scripts/fetch_data.py`` and is not
    committed here.

Both adapters write the same :class:`Material` records used by the synthetic
scenario.  Missing properties are stored as NaN; the trainer masks them.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .formula import element_roles, parse_formula
from .synth import Material, compute_descriptors

MATBENCH_REFERENCE = (
    "Dunn et al., npj Comput. Mater. 6, 138 (2020); "
    "data: Hacking Materials, figshare 10.6084/m9.figshare.9765779"
)
EES_REFERENCE = (
    "Haghshenas et al., EES Catalysis 2, 612-623 (2024), "
    "doi:10.1039/D3EY00246B; data: github.com/yousof96/ligh_intensity_paper"
)

_EES_COLUMNS = (
    "id",
    "paper",
    "description",
    "eg",
    "her",
    "cocatalyst",
    "cocatalyst_loading",
    "surface_area",
    "sacrificial_agent",
    "sacrificial_concentration",
    "light_intensity",
    "synthesis",
    "rate_unit",
)


def _real_material(
    identifier: str,
    formula: str,
    *,
    eg: float = math.nan,
    her: float = math.nan,
    stability: float = math.nan,
    cocatalyst: str = "none",
    cocatalyst_loading: float = 0.0,
    surface_area: float = 0.0,
    light_intensity: float = 0.0,
    catalyst_loading: float = 0.0,
    sacrificial_agent: str = "unknown",
    sacrificial_concentration: float = 0.0,
    method: str = "unknown",
    ph: float = 7.0,
    source: str = "external",
    reference: str = "",
) -> Material:
    elements = element_roles(formula)
    descriptors = compute_descriptors(
        elements,
        "unknown",
        ph,
        max(surface_area, 0.0),
        0.0,
        0.0,
    )
    return Material(
        id=identifier,
        formula=formula,
        family="unknown",
        crystal_system="unknown",
        elements=tuple(elements),
        dopant="none",
        dopant_fraction=0.0,
        cocatalyst=cocatalyst,
        method=method,
        ph=ph,
        surface_area=float(surface_area),
        strain=0.0,
        defect=0.0,
        heterojunction="none",
        eg=float(eg),
        her=float(her),
        stability=float(stability),
        cs_eff=math.nan,
        cb=math.nan,
        vb=math.nan,
        cocatalyst_loading=float(cocatalyst_loading),
        light_intensity=float(light_intensity),
        catalyst_loading=float(catalyst_loading),
        sacrificial_agent=sacrificial_agent,
        sacrificial_concentration=float(sacrificial_concentration),
        descriptors=descriptors,
        source=source,
        reference=reference,
    )


def load_matbench_gap(path: str | Path, limit: int | None = None) -> list[Material]:
    """Load compositions and experimental gaps from a Matbench JSON file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    columns = payload["columns"]
    composition_index = columns.index("composition")
    gap_index = columns.index("gap expt")
    materials: list[Material] = []
    skipped: list[str] = []
    for row in payload["data"]:
        formula = str(row[composition_index])
        try:
            parse_formula(formula)
        except ValueError:
            skipped.append(formula)
            continue
        gap = float(row[gap_index])
        if not np.isfinite(gap):
            skipped.append(formula)
            continue
        material = _real_material(
            f"mb-{len(materials) + 1:05d}",
            formula,
            eg=gap,
            source="matbench_expt_gap",
            reference=MATBENCH_REFERENCE,
        )
        materials.append(material)
        if limit and len(materials) >= limit:
            break
    if skipped:
        print(f"matbench: skipped {len(skipped)} invalid/empty rows (first: {skipped[:3]})")
    return materials


def _parse_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def load_ees_her(path: str | Path) -> list[Material]:
    """Load the EES TiO2 process records from the CSV produced by fetch_data.py."""
    path = Path(path)
    materials: list[Material] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(_EES_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"EES CSV is missing columns: {sorted(missing)}")
        for row in reader:
            her = _parse_float(row["her"])
            surface = _parse_float(row["surface_area"])
            light = _parse_float(row["light_intensity"])
            eg = _parse_float(row["eg"])
            loading = _parse_float(row["cocatalyst_loading"], 0.0)
            if not all(np.isfinite(v) for v in (her, surface, light, eg)):
                # Listwise deletion keeps the benchmark reproducible without
                # inventing missing experimental conditions.
                continue
            cocatalyst = str(row["cocatalyst"]).strip()
            if cocatalyst.lower() in ("bare", "none", ""):
                cocatalyst = "none"
            material = _real_material(
                f"ees-{row['id']}",
                "TiO2",
                eg=eg,
                her=her,
                cocatalyst=cocatalyst,
                cocatalyst_loading=loading,
                surface_area=surface,
                light_intensity=light,
                sacrificial_agent=str(row["sacrificial_agent"]).strip().lower() or "unknown",
                sacrificial_concentration=_parse_float(row["sacrificial_concentration"], 0.0),
                method=str(row["synthesis"]).strip().lower() or "unknown",
                source="ees_her",
                reference=f"{row['paper']}",
            )
            materials.append(material)
    return materials


def assign_random_splits(
    materials: Sequence[Material],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(materials))
    n_test = max(1, int(round(test_fraction * len(materials))))
    n_val = max(1, int(round(val_fraction * len(materials))))
    for position, index in enumerate(order):
        material = materials[int(index)]
        if position < n_test:
            material.split = "test"
        elif position < n_test + n_val:
            material.split = "val"
        else:
            material.split = "train"


def assign_group_splits(
    materials: Sequence[Material],
    key: Callable[[Material], object],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> None:
    """Split by a group key so no group appears in more than one split."""
    rng = np.random.default_rng(seed)
    groups: dict[object, list[Material]] = {}
    for material in materials:
        groups.setdefault(key(material), []).append(material)
    keys = list(groups)
    rng.shuffle(keys)
    n_test = max(1, int(round(test_fraction * len(keys))))
    n_val = max(1, int(round(val_fraction * len(keys))))
    for position, group_key in enumerate(keys):
        if position < n_test:
            split = "test"
        elif position < n_test + n_val:
            split = "val"
        else:
            split = "train"
        for material in groups[group_key]:
            material.split = split


def split_summary(materials: Sequence[Material]) -> dict[str, int]:
    return {
        split: sum(material.split == split for material in materials)
        for split in ("train", "val", "test")
    }
