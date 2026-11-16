"""SQLite-backed triple store plus a fast numeric property index."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from ..data.synth import Material
from .ontology import (
    Triple,
    design_target_entities,
    design_target_triples,
    entity_definitions,
    material_triples,
)


class KGStore:
    """A lightweight property graph.

    Two access paths are maintained: the semantic triple table (for paths and
    explanations) and a typed ``properties`` table (for fast numeric screening).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    # ------------------------------------------------------------------ schema
    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT DEFAULT 'structured'
            );
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL,
                value REAL,
                unit TEXT DEFAULT '',
                source TEXT DEFAULT 'structured',
                confidence REAL DEFAULT 1.0
            );
            CREATE TABLE IF NOT EXISTS properties (
                material_id TEXT PRIMARY KEY,
                formula TEXT NOT NULL,
                label TEXT NOT NULL,
                family TEXT NOT NULL,
                eg REAL NOT NULL,
                her REAL NOT NULL,
                stability REAL NOT NULL,
                cs_eff REAL NOT NULL,
                cb REAL NOT NULL,
                vb REAL NOT NULL,
                ph REAL NOT NULL,
                dopant TEXT NOT NULL,
                cocatalyst TEXT NOT NULL,
                synthesis_method TEXT NOT NULL,
                split TEXT DEFAULT '',
                source TEXT DEFAULT 'synthetic'
            );
            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
            CREATE INDEX IF NOT EXISTS idx_triples_relation ON triples(relation);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_sr ON triples(subject, relation);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_triples_unique
                ON triples(subject, relation, object, source);
            CREATE INDEX IF NOT EXISTS idx_properties_eg ON properties(eg);
            CREATE INDEX IF NOT EXISTS idx_properties_her ON properties(her);
            CREATE INDEX IF NOT EXISTS idx_properties_family ON properties(family);
            """
        )
        self.connection.commit()

    # ------------------------------------------------------------------- loading
    def add_entities(self, entities: Iterable[tuple[str, str, str]], source: str = "structured") -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO entities(id, name, type, source) VALUES (?, ?, ?, ?)",
            [(identifier, name, kind, source) for identifier, name, kind in entities],
        )

    def add_triples(self, triples: Iterable[Triple], source: str | None = None) -> None:
        self.connection.executemany(
            """INSERT OR IGNORE INTO triples(subject, relation, object, value, unit, source, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    triple.subject,
                    triple.relation,
                    triple.object,
                    triple.value,
                    triple.unit,
                    source or triple.source,
                    triple.confidence,
                )
                for triple in triples
            ],
        )

    def add_property(self, material: Material, source: str = "synthetic") -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO properties(
                   material_id, formula, label, family, eg, her, stability, cs_eff, cb, vb,
                   ph, dopant, cocatalyst, synthesis_method, split, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                material.id,
                material.formula,
                material.label,
                material.family,
                material.eg,
                material.her,
                material.stability,
                material.cs_eff,
                material.cb,
                material.vb,
                material.ph,
                material.dopant,
                material.cocatalyst,
                material.method,
                material.split,
                source,
            ),
        )

    def bulk_load(self, materials: Sequence[Material], source: str = "synthetic") -> None:
        """Populate entities, triples and the property index in one transaction."""
        seen_entities: set[str] = set()
        seen_triples: set[tuple[str, str, str]] = set()
        entity_rows: list[tuple[str, str, str, str]] = []
        triple_rows: list[tuple[object, ...]] = []
        property_rows: list[tuple[object, ...]] = []

        for material in materials:
            for identifier, name, kind in entity_definitions(material):
                if identifier in seen_entities:
                    continue
                seen_entities.add(identifier)
                entity_rows.append((identifier, name, kind, source))
            for triple in material_triples(material):
                key = triple.key()
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                triple_rows.append(
                    (
                        triple.subject,
                        triple.relation,
                        triple.object,
                        triple.value,
                        triple.unit,
                        triple.source,
                        triple.confidence,
                    )
                )
            property_rows.append(
                (
                    material.id,
                    material.formula,
                    material.label,
                    material.family,
                    material.eg,
                    material.her,
                    material.stability,
                    material.cs_eff,
                    material.cb,
                    material.vb,
                    material.ph,
                    material.dopant,
                    material.cocatalyst,
                    material.method,
                    material.split,
                    source,
                )
            )

        for identifier, name, kind in design_target_entities():
            if identifier not in seen_entities:
                seen_entities.add(identifier)
                entity_rows.append((identifier, name, kind, "design_target"))
        for triple in design_target_triples():
            if triple.key() not in seen_triples:
                triple_rows.append(
                    (
                        triple.subject,
                        triple.relation,
                        triple.object,
                        triple.value,
                        triple.unit,
                        triple.source,
                        triple.confidence,
                    )
                )

        self.connection.executemany(
            "INSERT OR REPLACE INTO entities(id, name, type, source) VALUES (?, ?, ?, ?)",
            entity_rows,
        )
        self.connection.executemany(
            """INSERT OR IGNORE INTO triples(subject, relation, object, value, unit, source, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            triple_rows,
        )
        self.connection.executemany(
            """INSERT OR REPLACE INTO properties(
                   material_id, formula, label, family, eg, her, stability, cs_eff, cb, vb,
                   ph, dopant, cocatalyst, synthesis_method, split, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            property_rows,
        )
        self.connection.commit()

    def remove_source(self, source: str) -> None:
        self.connection.execute("DELETE FROM triples WHERE source = ?", (source,))
        self.connection.commit()

    # ------------------------------------------------------------------- queries
    def stats(self) -> dict[str, int]:
        entities = self.connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        triples = self.connection.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        properties = self.connection.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
        relations = self.connection.execute(
            "SELECT COUNT(DISTINCT relation) FROM triples"
        ).fetchone()[0]
        return {
            "entities": int(entities),
            "triples": int(triples),
            "materials": int(properties),
            "relations": int(relations),
        }

    def query_materials(
        self,
        eg: tuple[float, float] | None = None,
        her_min: float | None = None,
        stability_min: float | None = None,
        families: Sequence[str] | None = None,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if eg is not None:
            clauses.append("eg BETWEEN ? AND ?")
            parameters.extend([eg[0], eg[1]])
        if her_min is not None:
            clauses.append("her >= ?")
            parameters.append(her_min)
        if stability_min is not None:
            clauses.append("stability >= ?")
            parameters.append(stability_min)
        if families:
            placeholders = ",".join("?" for _ in families)
            clauses.append(f"family IN ({placeholders})")
            parameters.extend(families)
        if split:
            clauses.append("split = ?")
            parameters.append(split)
        sql = "SELECT * FROM properties"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY her DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in self.connection.execute(sql, parameters)]

    def material(self, material_id: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT * FROM properties WHERE material_id = ?", (material_id,)
        ).fetchone()
        return dict(row) if row else None

    def triples(
        self,
        subject: str | None = None,
        relation: str | None = None,
        obj: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if subject is not None:
            clauses.append("subject = ?")
            parameters.append(subject)
        if relation is not None:
            clauses.append("relation = ?")
            parameters.append(relation)
        if obj is not None:
            clauses.append("object = ?")
            parameters.append(obj)
        sql = "SELECT subject, relation, object, value, unit, source FROM triples"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in self.connection.execute(sql, parameters)]

    def entity(self, identifier: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT id, name, type FROM entities WHERE id = ?", (identifier,)
        ).fetchone()
        return dict(row) if row else None

    def neighbors(
        self, identifier: str, relation: str | None = None, direction: str = "both", limit: int = 50
    ) -> list[dict[str, object]]:
        clauses = ["(subject = ? OR object = ?)"]
        parameters: list[object] = [identifier, identifier]
        if direction == "out":
            clauses = ["subject = ?"]
            parameters = [identifier]
        elif direction == "in":
            clauses = ["object = ?"]
            parameters = [identifier]
        if relation:
            clauses.append("relation = ?")
            parameters.append(relation)
        sql = (
            "SELECT subject, relation, object, value, unit, source FROM triples WHERE "
            + " AND ".join(clauses)
            + f" LIMIT {int(limit)}"
        )
        return [dict(row) for row in self.connection.execute(sql, parameters)]

    def find_paths(
        self, source: str, target: str, max_depth: int = 3, max_paths: int = 10
    ) -> list[list[tuple[str, str, str]]]:
        """Breadth-first paths of triples from source to target."""
        queue: list[tuple[str, list[tuple[str, str, str]], set[str]]] = [(source, [], {source})]
        paths: list[list[tuple[str, str, str]]] = []
        while queue and len(paths) < max_paths:
            node, path, visited = queue.pop(0)
            if len(path) >= max_depth:
                continue
            for row in self.connection.execute(
                "SELECT subject, relation, object FROM triples WHERE subject = ? LIMIT 60", (node,)
            ):
                next_node = row["object"]
                step = (row["subject"], row["relation"], next_node)
                if next_node in visited:
                    continue
                new_path = path + [step]
                if next_node == target:
                    paths.append(new_path)
                    if len(paths) >= max_paths:
                        break
                else:
                    queue.append((next_node, new_path, visited | {next_node}))
        return paths

    # ------------------------------------------------------------------ helpers
    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KGStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def build_kg(path: str | Path, materials: Sequence[Material]) -> KGStore:
    store = KGStore(path)
    store.bulk_load(materials, source="synthetic")
    return store
