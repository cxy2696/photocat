"""Chemical formula parsing and element-role assignment.

Matbench formulas contain nested parentheses and fractional stoichiometries
(for example ``Ba2Sb7HO14`` or ``Ag(AuS)2``).  The parser here is small but
handles that grammar and is used by the real-data adapters.  It deliberately
rejects charges, hydrates and labels such as ``G1128`` that are not valid
element compositions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .elements import BY_SYMBOL

_TOKEN = re.compile(r"[A-Z][a-z]?|\(|\)|(?:\d+\.?\d*|\.\d+)")
_ANIONS = frozenset({"O", "N", "S", "Se", "Te", "F", "Cl", "Br", "I"})


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a chemical formula into element -> stoichiometry.

    Raises ``ValueError`` for unsupported or non-chemical strings.
    """
    cleaned = formula.replace(" ", "").replace("−", "-")
    if cleaned.startswith("g-"):
        cleaned = cleaned[2:]
    tokens = _TOKEN.findall(cleaned)
    if not tokens or "".join(tokens) != cleaned:
        raise ValueError(f"unsupported formula: {formula!r}")
    counts, position = _parse_group(tokens, 0)
    if position != len(tokens):
        raise ValueError(f"unbalanced formula: {formula!r}")
    if not counts:
        raise ValueError(f"empty formula: {formula!r}")
    unknown = sorted(set(counts) - set(BY_SYMBOL))
    if unknown:
        raise ValueError(f"unknown elements {unknown} in {formula!r}")
    return dict(counts)


def _parse_group(tokens: list[str], start: int) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    position = start
    while position < len(tokens):
        token = tokens[position]
        if token == ")":
            return counts, position
        if token == "(":
            nested, position = _parse_group(tokens, position + 1)
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unbalanced parentheses")
            position += 1
            multiplier, position = _number(tokens, position)
            for symbol, count in nested.items():
                counts[symbol] += count * multiplier
            continue
        if not re.fullmatch(r"[A-Z][a-z]?", token):
            raise ValueError(f"unexpected token {token!r}")
        position += 1
        multiplier, position = _number(tokens, position)
        counts[token] += multiplier
    return counts, position


def _number(tokens: list[str], position: int) -> tuple[float, int]:
    if position < len(tokens) and re.fullmatch(r"\d+\.?\d*|\.\d+", tokens[position]):
        return float(tokens[position]), position + 1
    return 1.0, position


def element_roles(formula: str) -> list[tuple[str, float, str]]:
    """Return ``(symbol, stoichiometry, role)`` with cathode/anode-style roles.

    Chalcogens and halogens are treated as anions, phosphorus as an anion only
    in phosphides (no oxygen present), and everything else as a cation.  This
    is a composition heuristic, not an oxidation-state assignment.
    """
    counts = parse_formula(formula)
    has_oxygen = "O" in counts
    roles: list[tuple[str, float, str]] = []
    for symbol, count in counts.items():
        if symbol in _ANIONS:
            role = "anion"
        elif symbol == "P" and not has_oxygen:
            role = "anion"
        elif symbol == "H":
            role = "cation"
        else:
            role = "cation"
        roles.append((symbol, float(count), role))
    if not any(role == "cation" for _, _, role in roles):
        # e.g. N2O or interhalogens: use the least electronegative atom as cation.
        candidates = sorted(roles, key=lambda item: BY_SYMBOL[item[0]].chi)
        symbol, count, _ = candidates[0]
        roles = [
            (s, q, "cation" if s == symbol else "anion")
            for s, q, _ in roles
        ]
    return roles


def formula_from_roles(elements: Iterable[tuple[str, float, str]]) -> str:
    """Canonical display formula from an element list (no fractional reduction)."""
    parts = []
    for symbol, count, _ in elements:
        parts.append(symbol if abs(count - 1.0) < 1e-9 else f"{symbol}{count:g}")
    return "".join(parts)
