"""Data layer: periodic table, synthetic library, featurisation and I/O."""

from .elements import BY_SYMBOL, ELEMENT_IDS, ELEMENT_LIST, ELEMENT_FEATURES, MATERIAL_FEATURES
from .synth import (
    FAMILIES,
    HOLDOUT_FAMILIES,
    DatasetBundle,
    Material,
    assign_splits,
    generate_dataset,
    generate_materials,
    load_materials,
    make_anchors,
    parse_formula,
    save_materials,
    summary,
)

__all__ = [
    "BY_SYMBOL",
    "ELEMENT_IDS",
    "ELEMENT_LIST",
    "ELEMENT_FEATURES",
    "MATERIAL_FEATURES",
    "FAMILIES",
    "HOLDOUT_FAMILIES",
    "DatasetBundle",
    "Material",
    "assign_splits",
    "generate_dataset",
    "generate_materials",
    "load_materials",
    "make_anchors",
    "parse_formula",
    "save_materials",
    "summary",
]
