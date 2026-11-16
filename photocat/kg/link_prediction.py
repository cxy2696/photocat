"""DistMult link prediction for KG completion.

The proposal cites graph-representation learning for KG completion.  We train a
DistMult model with negative sampling on the categorical part of the graph
(elements, family, crystal system, co-catalyst, method, dopant) and report
filtered MRR / Hits@10.  Numeric property edges are deliberately excluded so
the task is genuine relational completion rather than value regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..autograd import Adam, Tensor, bce_with_logits, embedding_lookup
from .store import KGStore

LINK_RELATIONS: tuple[str, ...] = (
    "hasElement",
    "hasFamily",
    "hasCrystalSystem",
    "hasCoCatalyst",
    "hasSynthesisMethod",
    "hasDopant",
)


@dataclass
class LinkPredictionResult:
    metrics: dict[str, float] = field(default_factory=dict)
    examples: list[dict[str, object]] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0


def _load_triples(store: KGStore) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    query = (
        "SELECT subject, relation, object FROM triples WHERE relation IN ("
        + ",".join("?" for _ in LINK_RELATIONS)
        + ") AND source = 'structured'"
    )
    rows = list(store.connection.execute(query, LINK_RELATIONS))
    subjects = [row["subject"] for row in rows]
    relations = [row["relation"] for row in rows]
    objects = [row["object"] for row in rows]
    return (
        subjects,
        relations,
        objects,
        sorted(set(subjects) | set(objects)),
        sorted(set(relations)),
    )


def _index(values: Sequence[str], vocab: dict[str, int]) -> np.ndarray:
    return np.asarray([vocab[value] for value in values], dtype=np.int64)


def _entity_vocab(subjects: Sequence[str], objects: Sequence[str]) -> dict[str, int]:
    return {value: i for i, value in enumerate(sorted(set(subjects) | set(objects)))}


def _filtered_metrics(
    scores: np.ndarray,
    true_index: int,
    candidates: Sequence[str],
    true_set: set[str],
) -> tuple[float, int]:
    """Return (reciprocal rank, rank) after removing other true objects."""
    order = np.argsort(-scores)
    rank = 1
    for index in order:
        candidate = candidates[int(index)]
        if candidate == candidates[true_index]:
            return 1.0 / rank, rank
        if candidate not in true_set:
            rank += 1
    return 0.0, rank


def train_link_predictor(
    store: KGStore,
    dim: int = 32,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 0.01,
    n_negatives: int = 8,
    seed: int = 0,
    full_vocab_eval: int = 300,
) -> LinkPredictionResult:
    subjects, relations, objects, entity_list, relation_list = _load_triples(store)
    if not subjects:
        return LinkPredictionResult(metrics={"status": "no triples"})
    entity_vocab = _entity_vocab(subjects, objects)
    relation_vocab = {relation: i for i, relation in enumerate(relation_list)}
    s = _index(subjects, entity_vocab)
    r = _index(relations, relation_vocab)
    o = _index(objects, entity_vocab)
    n = len(s)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    n_test = max(1, int(0.1 * n))
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    n_entities = len(entity_vocab)
    n_relations = len(relation_list)

    entity_embedding = Tensor(rng.normal(0.0, 0.1, (n_entities, dim)), requires_grad=True)
    relation_embedding = Tensor(rng.normal(0.0, 0.1, (n_relations, dim)), requires_grad=True)
    optimizer = Adam([entity_embedding, relation_embedding], lr=lr, weight_decay=1e-6)

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_idx)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start : start + batch_size]
            negative_objects = rng.integers(0, n_entities, size=(len(batch), n_negatives))
            batch_s = np.concatenate([s[batch], np.repeat(s[batch], n_negatives)])
            batch_r = np.concatenate([r[batch], np.repeat(r[batch], n_negatives)])
            batch_o = np.concatenate([o[batch], negative_objects.reshape(-1)])
            labels = np.concatenate(
                [np.ones(len(batch)), np.zeros(len(batch) * n_negatives)]
            )
            subject_vec = embedding_lookup(entity_embedding, batch_s)
            relation_vec = embedding_lookup(relation_embedding, batch_r)
            object_vec = embedding_lookup(entity_embedding, batch_o)
            scores = (subject_vec * relation_vec * object_vec).sum(axis=-1)
            loss = bce_with_logits(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.clip_grad_norm(5.0)
            optimizer.step()
            total_loss += float(loss.data)
            batches += 1
        print(
            f"link-pred epoch {epoch:2d} | loss {total_loss / max(batches, 1):.4f}",
            flush=True,
        )

    # Relation-constrained filtered ranking.
    candidates_by_relation: dict[int, list[str]] = {}
    for relation_index in range(n_relations):
        candidates_by_relation[relation_index] = sorted(
            {objects[i] for i in np.where(r == relation_index)[0]}
        )
    true_by_pair: dict[tuple[int, int], set[str]] = {}
    for i in range(n):
        true_by_pair.setdefault((int(s[i]), int(r[i])), set()).add(objects[i])

    reciprocal_ranks: list[float] = []
    hits1 = hits10 = 0
    for index in test_idx:
        s_i, r_i, o_i = int(s[index]), int(r[index]), int(o[index])
        candidates = candidates_by_relation[r_i]
        if objects[o_i] not in candidates:
            continue
        candidate_ids = np.asarray([entity_vocab[c] for c in candidates], dtype=np.int64)
        subj = entity_embedding.data[s_i]
        rel = relation_embedding.data[r_i]
        scores = (entity_embedding.data[candidate_ids] * (subj * rel)).sum(axis=1)
        true_position = candidates.index(objects[o_i])
        true_set = true_by_pair[(s_i, r_i)] - {objects[o_i]}
        rr, rank = _filtered_metrics(scores, true_position, candidates, true_set)
        reciprocal_ranks.append(rr)
        hits1 += int(rank == 1)
        hits10 += int(rank <= 10)

    metrics = {
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hits@1": float(hits1 / max(len(reciprocal_ranks), 1)),
        "hits@10": float(hits10 / max(len(reciprocal_ranks), 1)),
        "test_triples": float(len(reciprocal_ranks)),
        "train_triples": float(len(train_idx)),
        "entities": float(n_entities),
        "relations": float(n_relations),
    }

    # Full-vocabulary ranking on a small sample (harder, more realistic).
    if full_vocab_eval > 0:
        sample = rng.choice(test_idx, min(full_vocab_eval, len(test_idx)), replace=False)
        rr_all: list[float] = []
        for index in sample:
            s_i, r_i, o_i = int(s[index]), int(r[index]), int(o[index])
            subj = entity_embedding.data[s_i]
            rel = relation_embedding.data[r_i]
            scores = (entity_embedding.data * (subj * rel)).sum(axis=1)
            true_name = objects[o_i]
            true_set = true_by_pair[(s_i, r_i)] - {true_name}
            order = np.argsort(-scores)
            rank = 1
            for candidate_id in order:
                candidate = entity_list[int(candidate_id)]
                if candidate == true_name:
                    break
                if candidate not in true_set:
                    rank += 1
            rr_all.append(1.0 / rank)
        metrics["full_vocab_mrr_sample"] = float(np.mean(rr_all)) if rr_all else 0.0
        metrics["full_vocab_sample_size"] = float(len(rr_all))

    # Qualitative completions: score methods / co-catalysts for a few materials.
    examples: list[dict[str, object]] = []
    relation_lookup = {value: key for key, value in relation_vocab.items()}
    for material_index in rng.choice(test_idx, min(5, len(test_idx)), replace=False):
        s_i, r_i = int(s[material_index]), int(r[material_index])
        candidates = candidates_by_relation[r_i]
        candidate_ids = np.asarray([entity_vocab[c] for c in candidates], dtype=np.int64)
        scores = (
            entity_embedding.data[candidate_ids]
            * (entity_embedding.data[s_i] * relation_embedding.data[r_i])
        ).sum(axis=1)
        top = np.argsort(-scores)[:3]
        examples.append(
            {
                "subject": subjects[int(material_index)],
                "relation": relation_lookup[r_i],
                "true": objects[int(material_index)],
                "predicted": [
                    {"object": candidates[int(i)], "score": round(float(scores[int(i)]), 4)}
                    for i in top
                ],
            }
        )
    return LinkPredictionResult(
        metrics=metrics,
        examples=examples,
        n_train=len(train_idx),
        n_test=len(test_idx),
    )


def save_link_prediction(path: str | Path, result: LinkPredictionResult) -> None:
    Path(path).write_text(
        json.dumps(
            {"metrics": result.metrics, "examples": result.examples},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
