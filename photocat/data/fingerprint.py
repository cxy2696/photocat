"""Composition fingerprints and Tanimoto novelty.

A Morgan-style circular fingerprint would need RDKit; photocatalysts are mostly
inorganic, so we use a hashed composition/structure fingerprint: element
presence, stoichiometry bins, and element-pair co-occurrence.  Tanimoto
similarity over these bits is the novelty proxy used by the screening stage
(the proposal's ``Tanimoto < 0.8`` criterion).
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .synth import Material

DEFAULT_BITS = 512


def _bit(token: str, n_bits: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_bits


def composition_fingerprint(material: Material, n_bits: int = DEFAULT_BITS) -> np.ndarray:
    total = sum(stoich for _, stoich, _ in material.elements) or 1.0
    bits: set[int] = set()
    fractions: list[tuple[str, float]] = []
    for symbol, stoich, role in material.elements:
        fraction = stoich / total
        fractions.append((symbol, fraction))
        bits.add(_bit(f"element:{symbol}", n_bits))
        bits.add(_bit(f"fraction:{symbol}:{min(int(fraction * 10), 9)}", n_bits))
        bits.add(_bit(f"role:{symbol}:{role}", n_bits))
    for i in range(len(fractions)):
        for j in range(i + 1, len(fractions)):
            first, second = sorted((fractions[i][0], fractions[j][0]))
            bits.add(_bit(f"pair:{first}-{second}", n_bits))
            ratio = min(int((fractions[i][1] / max(fractions[j][1], 1e-6)) * 5), 20)
            bits.add(_bit(f"ratio:{first}-{second}:{ratio}", n_bits))
    vector = np.zeros(n_bits, dtype=np.float32)
    vector[list(bits)] = 1.0
    return vector


def fingerprint_matrix(
    materials: Sequence[Material], n_bits: int = DEFAULT_BITS
) -> np.ndarray:
    if not materials:
        return np.zeros((0, n_bits), dtype=np.float32)
    matrix = np.stack([composition_fingerprint(material, n_bits) for material in materials])
    return matrix.astype(np.float32)


def tanimoto_matrix(query: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto similarity between binary fingerprint matrices."""
    query = query.astype(np.float32)
    library = library.astype(np.float32)
    intersection = query @ library.T
    query_counts = query.sum(axis=1, keepdims=True)
    library_counts = library.sum(axis=1)[None, :]
    union = query_counts + library_counts - intersection
    return intersection / np.maximum(union, 1e-9)


def max_similarity(
    query: np.ndarray,
    library: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Max Tanimoto similarity of each query to a library, in chunks."""
    maxima = np.zeros(len(query), dtype=np.float32)
    nearest = np.zeros(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        block = tanimoto_matrix(query[start : start + chunk_size], library)
        maxima[start : start + chunk_size] = block.max(axis=1)
        nearest[start : start + chunk_size] = block.argmax(axis=1)
    return maxima, nearest


def pairwise_similarity(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float(np.dot(a, b))
    union = float(a.sum() + b.sum() - intersection)
    return intersection / max(union, 1e-9)


def diversity_select(
    scores: np.ndarray,
    fingerprints: np.ndarray,
    k: int,
    max_similarity: float = 0.92,
) -> list[int]:
    """Greedy score-then-diversity selection (max-min similarity threshold)."""
    order = np.argsort(-scores)
    selected: list[int] = []
    for index in order:
        if len(selected) >= k:
            break
        if all(
            pairwise_similarity(fingerprints[index], fingerprints[other]) < max_similarity
            for other in selected
        ):
            selected.append(int(index))
    if len(selected) < k:
        for index in order:
            if len(selected) >= k:
                break
            if int(index) not in selected:
                selected.append(int(index))
    return selected
