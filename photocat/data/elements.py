"""A compact periodic-table knowledge base for photocatalyst chemistry.

Only elements that appear in common oxide, oxynitride, sulfide, halide and
carbon-nitride photocatalysts are included.  Values are rounded literature
values intended for featurisation, not for thermodynamic calculations:
Pauling electronegativity, Shannon ionic radius (6-coordinate), common
oxidation state, crustal abundance (ppm), and nominal d-electron count.

The ``eg_offset`` term is a coarse chemical prior used by the synthetic data
generator to reproduce well-known trends (for example SnO2 and ZrO2 having
wider gaps than TiO2).  It is *not* used as a model feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements_extra import EXTRA_ROWS


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    z: int
    chi: float
    radius: float
    valence: int
    mass: float
    abundance: float  # crustal abundance, ppm
    d_electrons: int
    group: int
    period: int
    eg_offset: float = 0.0
    anion_type: str | None = None

    @property
    def is_anion(self) -> bool:
        return self.anion_type is not None


_ELEMENTS: tuple[Element, ...] = (
    # --- anions / framework anions -----------------------------------------
    Element("O", "oxygen", 8, 3.44, 1.40, -2, 16.00, 461000.0, 0, 16, 2, 0.0, "O"),
    Element("N", "nitrogen", 7, 3.04, 1.46, -3, 14.01, 19.0, 0, 15, 2, 0.0, "N"),
    Element("S", "sulfur", 16, 2.58, 1.84, -2, 32.06, 350.0, 0, 16, 3, 0.0, "S"),
    Element("F", "fluorine", 9, 3.98, 1.33, -1, 19.00, 585.0, 0, 17, 2, 0.0, "F"),
    Element("Cl", "chlorine", 17, 3.16, 1.81, -1, 35.45, 145.0, 0, 17, 3, 0.0, "Cl"),
    Element("Br", "bromine", 35, 2.96, 1.96, -1, 79.90, 2.4, 0, 17, 4, 0.0, "Br"),
    Element("I", "iodine", 53, 2.66, 2.20, -1, 126.90, 0.45, 0, 17, 5, 0.0, "I"),
    # --- cations ------------------------------------------------------------
    Element("Ti", "titanium", 22, 1.54, 0.605, 4, 47.87, 5650.0, 0, 4, 4, 0.00),
    Element("Zr", "zirconium", 40, 1.33, 0.720, 4, 91.22, 165.0, 0, 4, 5, 0.80),
    Element("Sn", "tin", 50, 1.96, 0.690, 4, 118.71, 2.3, 10, 14, 5, 0.35),
    Element("Ce", "cerium", 58, 1.12, 0.870, 4, 140.12, 66.5, 0, 3, 6, 0.00),
    Element("V", "vanadium", 23, 1.63, 0.640, 5, 50.94, 120.0, 0, 5, 4, 0.20),
    Element("Nb", "niobium", 41, 1.60, 0.640, 5, 92.91, 20.0, 0, 5, 5, 0.15),
    Element("Ta", "tantalum", 73, 1.50, 0.640, 5, 180.95, 2.0, 0, 5, 6, 0.15),
    Element("W", "tungsten", 74, 2.36, 0.600, 6, 183.84, 1.25, 0, 6, 6, 0.00),
    Element("Mo", "molybdenum", 42, 2.16, 0.650, 6, 95.95, 1.2, 0, 6, 5, 0.00),
    Element("Bi", "bismuth", 83, 2.02, 1.030, 3, 208.98, 0.0085, 10, 15, 6, 0.00),
    Element("Fe", "iron", 26, 1.83, 0.645, 3, 55.85, 56300.0, 5, 8, 4, -0.05),
    Element("Zn", "zinc", 30, 1.65, 0.740, 2, 65.38, 70.0, 10, 12, 4, 0.30),
    Element("Cu", "copper", 29, 1.90, 0.730, 2, 63.55, 60.0, 9, 11, 4, -0.05),
    Element("Ni", "nickel", 28, 1.91, 0.690, 2, 58.69, 84.0, 8, 10, 4, 0.00),
    Element("Co", "cobalt", 27, 1.88, 0.745, 2, 58.93, 25.0, 7, 9, 4, -0.05),
    Element("Mn", "manganese", 25, 1.55, 0.830, 2, 54.94, 950.0, 5, 7, 4, -0.10),
    Element("Cr", "chromium", 24, 1.66, 0.615, 3, 52.00, 102.0, 3, 6, 4, 0.20),
    Element("Al", "aluminium", 13, 1.61, 0.535, 3, 26.98, 82300.0, 0, 13, 3, 0.90),
    Element("Ga", "gallium", 31, 1.81, 0.620, 3, 69.72, 19.0, 10, 13, 4, 0.30),
    Element("In", "indium", 49, 1.78, 0.800, 3, 114.82, 0.25, 10, 13, 5, 0.20),
    Element("La", "lanthanum", 57, 1.10, 1.030, 3, 138.91, 39.0, 0, 3, 6, 0.20),
    Element("Ca", "calcium", 20, 1.00, 1.000, 2, 40.08, 41500.0, 0, 2, 4, 0.10),
    Element("Sr", "strontium", 38, 0.95, 1.180, 2, 87.62, 370.0, 0, 2, 5, 0.10),
    Element("Ba", "barium", 56, 0.89, 1.350, 2, 137.33, 425.0, 0, 2, 6, 0.15),
    Element("Mg", "magnesium", 12, 1.31, 0.720, 2, 24.31, 23300.0, 0, 2, 3, 0.50),
    Element("K", "potassium", 19, 0.82, 1.380, 1, 39.10, 20900.0, 0, 1, 4, 0.10),
    Element("Na", "sodium", 11, 0.93, 1.020, 1, 22.99, 23600.0, 0, 1, 3, 0.10),
    Element("Ag", "silver", 47, 1.93, 1.150, 1, 107.87, 0.075, 10, 11, 5, 0.20),
    Element("Cd", "cadmium", 48, 1.69, 0.950, 2, 112.41, 0.15, 10, 12, 5, 0.20),
    Element("P", "phosphorus", 15, 2.19, 0.440, 5, 30.97, 1050.0, 0, 15, 3, 0.00),
    Element("Si", "silicon", 14, 1.90, 0.400, 4, 28.09, 282000.0, 0, 14, 3, 0.00),
    Element("Y", "yttrium", 39, 1.22, 0.900, 3, 88.91, 33.0, 0, 3, 5, 0.20),
    Element("C", "carbon", 6, 2.55, 0.770, 4, 12.01, 200.0, 0, 14, 2, 0.35),
)

ELEMENT_LIST: tuple[Element, ...] = tuple(
    sorted((*_ELEMENTS, *(Element(*row) for row in EXTRA_ROWS)), key=lambda e: e.z)
)
BY_SYMBOL: dict[str, Element] = {element.symbol: element for element in ELEMENT_LIST}

#: Integer ids used by embeddings; 0 is reserved for padding.
ELEMENT_IDS: dict[str, int] = {element.symbol: i + 1 for i, element in enumerate(ELEMENT_LIST)}

DOPANT_LIBRARY: tuple[str, ...] = ("N", "S", "P", "K", "Fe", "Cu", "Zn", "C")

#: Nominal effect of a dopant on the gap, charge separation and stability.
DOPANT_EFFECTS: dict[str, tuple[float, float, float]] = {
    "N": (-0.35, 0.10, -0.05),
    "S": (-0.25, 0.12, -0.10),
    "P": (-0.15, 0.08, -0.02),
    "K": (-0.05, 0.05, 0.02),
    "Fe": (-0.30, 0.15, -0.08),
    "Cu": (-0.20, 0.10, -0.12),
    "Zn": (-0.10, 0.05, 0.04),
    "C": (-0.20, 0.06, 0.00),
}

COCATALYSTS: tuple[str, ...] = ("none", "Pt", "MoS2", "NiS", "Co-Pi", "Ni2P")
COCATALYST_FACTOR: dict[str, float] = {
    "none": 1.0,
    "Pt": 2.2,
    "MoS2": 1.8,
    "NiS": 1.6,
    "Co-Pi": 1.4,
    "Ni2P": 1.7,
}

METHODS: tuple[str, ...] = (
    "hydrothermal",
    "sol-gel",
    "solid-state",
    "CVD",
    "thermal-polymerization",
)

ANION_EG_OFFSET: dict[str, float] = {
    "O": 0.0,
    "N": -0.15,
    "S": -0.20,
    "F": 0.60,
    "Cl": 0.45,
    "Br": 0.15,
    "I": -0.85,
}

#: Feature layout for element nodes in the knowledge graph.
ELEMENT_FEATURES: tuple[str, ...] = (
    "chi",
    "radius",
    "valence",
    "log_mass",
    "log_abundance",
    "d_electrons",
    "group",
    "period",
    "is_anion",
)

#: Feature layout for the material node.
MATERIAL_FEATURES: tuple[str, ...] = (
    "chi_cation",
    "radius_cation",
    "radius_std",
    "d_cation",
    "log_abundance_min",
    "chi_anion",
    "n_elements",
    "distortion",
    "valence_cation",
)


def element_feature_matrix() -> np.ndarray:
    """Raw (V+1, F) feature matrix; row 0 is the padding row."""
    rows = [np.zeros(len(ELEMENT_FEATURES), dtype=np.float64)]
    for element in ELEMENT_LIST:
        rows.append(
            np.array(
                [
                    element.chi,
                    element.radius,
                    element.valence,
                    np.log1p(element.mass),
                    np.log10(element.abundance + 1.0),
                    element.d_electrons / 10.0,
                    element.group / 18.0,
                    element.period / 7.0,
                    1.0 if element.is_anion else 0.0,
                ],
                dtype=np.float64,
            )
        )
    return np.vstack(rows)


def standardise_element_features() -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature mean/std computed over the non-padding rows."""
    raw = element_feature_matrix()[1:]
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std
