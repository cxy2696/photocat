"""Additional periodic-table entries used to parse arbitrary formulas.

Generated from pymatgen 2026.10.24 periodic-table data.  Only elements not
already hand-curated in ``elements.py`` are listed.  Radii are average ionic
radii where available and atomic radii otherwise; valences are the most common
positive oxidation state (most negative for halogens/chalcogens); d-electron
count is left at zero and crustal abundance at a 10 ppm placeholder because the
upstream table does not provide oxidation-state-resolved values.  These entries
extend formula coverage for real datasets; they are not used by the synthetic
generator.
"""

from __future__ import annotations

# (symbol, name, Z, electronegativity, radius, valence, mass, abundance_ppm,
#  d_electrons, group, period, eg_offset, anion_type)
EXTRA_ROWS: tuple[tuple[object, ...], ...] = (
    ('H', 'Hydrogen', 1, 2.200, 0.250, 1, 1.008, 10.0, 0, 1, 1, 0.00, None),
    ('He', 'Helium', 2, 1.500, 1.000, 0, 4.003, 10.0, 0, 18, 1, 0.00, None),
    ('Li', 'Lithium', 3, 0.980, 0.900, 1, 6.941, 10.0, 0, 1, 2, 0.00, None),
    ('Be', 'Beryllium', 4, 1.570, 0.590, 2, 9.012, 10.0, 0, 2, 2, 0.00, None),
    ('B', 'Boron', 5, 2.040, 0.410, 3, 10.811, 10.0, 0, 13, 2, 0.00, None),
    ('Ne', 'Neon', 10, 1.500, 1.000, 0, 20.180, 10.0, 0, 18, 2, 0.00, None),
    ('Ar', 'Argon', 18, 1.500, 0.710, 0, 39.948, 10.0, 0, 18, 3, 0.00, None),
    ('Sc', 'Scandium', 21, 1.360, 0.885, 3, 44.956, 10.0, 0, 3, 4, 0.00, None),
    ('Ge', 'Germanium', 32, 2.010, 0.770, 4, 72.640, 10.0, 0, 14, 4, 0.00, None),
    ('As', 'Arsenic', 33, 2.180, 0.660, 5, 74.922, 10.0, 0, 15, 4, 0.00, None),
    ('Se', 'Selenium', 34, 2.550, 1.013, -2, 78.960, 10.0, 0, 16, 4, 0.00, "Se"),
    ('Kr', 'Krypton', 36, 3.000, 1.000, 2, 83.798, 10.0, 0, 18, 4, 0.00, None),
    ('Rb', 'Rubidium', 37, 0.820, 1.660, 1, 85.468, 10.0, 0, 1, 5, 0.00, None),
    ('Tc', 'Technetium', 43, 1.900, 0.742, 7, 98.000, 10.0, 0, 7, 5, 0.00, None),
    ('Ru', 'Ruthenium', 44, 2.200, 0.661, 4, 101.070, 10.0, 0, 8, 5, 0.00, None),
    ('Rh', 'Rhodium', 45, 2.280, 0.745, 3, 102.906, 10.0, 0, 9, 5, 0.00, None),
    ('Pd', 'Palladium', 46, 2.200, 0.846, 4, 106.420, 10.0, 0, 10, 5, 0.00, None),
    ('Sb', 'Antimony', 51, 2.050, 0.830, 5, 121.760, 10.0, 0, 15, 5, 0.00, None),
    ('Te', 'Tellurium', 52, 2.100, 1.293, -2, 127.600, 10.0, 0, 16, 5, 0.00, "Te"),
    ('Xe', 'Xenon', 54, 2.600, 0.620, 6, 131.293, 10.0, 0, 18, 5, 0.00, None),
    ('Cs', 'Cesium', 55, 0.790, 1.810, 1, 132.905, 10.0, 0, 1, 6, 0.00, None),
    ('Pr', 'Praseodymium', 59, 1.130, 1.060, 3, 140.908, 10.0, 0, 3, 6, 0.00, None),
    ('Nd', 'Neodymium', 60, 1.140, 1.276, 3, 144.242, 10.0, 0, 3, 6, 0.00, None),
    ('Pm', 'Promethium', 61, 1.130, 1.110, 3, 145.000, 10.0, 0, 3, 6, 0.00, None),
    ('Sm', 'Samarium', 62, 1.170, 1.229, 3, 150.360, 10.0, 0, 3, 6, 0.00, None),
    ('Eu', 'Europium', 63, 1.200, 1.199, 3, 151.964, 10.0, 0, 3, 6, 0.00, None),
    ('Gd', 'Gadolinium', 64, 1.200, 1.075, 3, 157.250, 10.0, 0, 3, 6, 0.00, None),
    ('Tb', 'Terbium', 65, 1.100, 0.982, 3, 158.925, 10.0, 0, 3, 6, 0.00, None),
    ('Dy', 'Dysprosium', 66, 1.220, 1.131, 3, 162.500, 10.0, 0, 3, 6, 0.00, None),
    ('Ho', 'Holmium', 67, 1.230, 1.041, 3, 164.930, 10.0, 0, 3, 6, 0.00, None),
    ('Er', 'Erbium', 68, 1.240, 1.030, 3, 167.259, 10.0, 0, 3, 6, 0.00, None),
    ('Tm', 'Thulium', 69, 1.250, 1.095, 3, 168.934, 10.0, 0, 3, 6, 0.00, None),
    ('Yb', 'Ytterbium', 70, 1.100, 1.084, 3, 173.040, 10.0, 0, 3, 6, 0.00, None),
    ('Lu', 'Lutetium', 71, 1.270, 1.001, 3, 174.967, 10.0, 0, 3, 6, 0.00, None),
    ('Hf', 'Hafnium', 72, 1.300, 0.850, 4, 178.490, 10.0, 0, 4, 6, 0.00, None),
    ('Re', 'Rhenium', 75, 1.900, 0.713, 7, 186.207, 10.0, 0, 7, 6, 0.00, None),
    ('Os', 'Osmium', 76, 2.200, 0.673, 4, 190.230, 10.0, 0, 8, 6, 0.00, None),
    ('Ir', 'Iridium', 77, 2.200, 0.765, 4, 192.217, 10.0, 0, 9, 6, 0.00, None),
    ('Pt', 'Platinum', 78, 2.280, 0.805, 4, 195.084, 10.0, 0, 10, 6, 0.00, None),
    ('Au', 'Gold', 79, 2.540, 1.070, 3, 196.967, 10.0, 0, 11, 6, 0.00, None),
    ('Hg', 'Mercury', 80, 2.000, 1.245, 2, 200.590, 10.0, 0, 12, 6, 0.00, None),
    ('Tl', 'Thallium', 81, 1.620, 1.333, 3, 204.383, 10.0, 0, 13, 6, 0.00, None),
    ('Pb', 'Lead', 82, 2.330, 1.123, 4, 207.200, 10.0, 0, 14, 6, 0.00, None),
    ('Po', 'Polonium', 84, 2.000, 0.945, 4, 210.000, 10.0, 0, 16, 6, 0.00, None),
    ('At', 'Astatine', 85, 2.200, 0.760, 1, 210.000, 10.0, 0, 17, 6, 0.00, None),
    ('Rn', 'Radon', 86, 2.200, 1.000, 0, 220.000, 10.0, 0, 18, 6, 0.00, None),
    ('Fr', 'Francium', 87, 0.700, 1.940, 1, 223.000, 10.0, 0, 1, 7, 0.00, None),
    ('Ra', 'Radium', 88, 0.900, 1.620, 2, 226.000, 10.0, 0, 2, 7, 0.00, None),
    ('Ac', 'Actinium', 89, 1.100, 1.260, 3, 227.000, 10.0, 0, 3, 7, 0.00, None),
    ('Th', 'Thorium', 90, 1.300, 1.080, 4, 232.038, 10.0, 0, 3, 7, 0.00, None),
    ('Pa', 'Protactinium', 91, 1.500, 1.040, 5, 231.036, 10.0, 0, 3, 7, 0.00, None),
    ('U', 'Uranium', 92, 1.380, 0.991, 6, 238.029, 10.0, 0, 3, 7, 0.00, None),
    ('Np', 'Neptunium', 93, 1.360, 1.000, 5, 237.000, 10.0, 0, 3, 7, 0.00, None),
    ('Pu', 'Plutonium', 94, 1.280, 0.968, 4, 244.000, 10.0, 0, 3, 7, 0.00, None),
    ('Am', 'Americium', 95, 1.300, 1.168, 3, 243.000, 10.0, 0, 3, 7, 0.00, None),
    ('Cm', 'Curium', 96, 1.300, 1.050, 3, 247.000, 10.0, 0, 3, 7, 0.00, None),
    ('Bk', 'Berkelium', 97, 1.300, 1.035, 3, 247.000, 10.0, 0, 3, 7, 0.00, None),
    ('Cf', 'Californium', 98, 1.300, 1.026, 3, 251.000, 10.0, 0, 3, 7, 0.00, None),
    ('Es', 'Einsteinium', 99, 1.300, 1.000, 3, 252.000, 10.0, 0, 3, 7, 0.00, None),
    ('Fm', 'Fermium', 100, 1.300, 1.000, 3, 257.000, 10.0, 0, 3, 7, 0.00, None),
    ('Md', 'Mendelevium', 101, 1.300, 1.000, 3, 258.000, 10.0, 0, 3, 7, 0.00, None),
    ('No', 'Nobelium', 102, 1.300, 1.000, 3, 259.000, 10.0, 0, 3, 7, 0.00, None),
    ('Lr', 'Lawrencium', 103, 1.300, 1.000, 3, 262.000, 10.0, 0, 3, 7, 0.00, None),
)
