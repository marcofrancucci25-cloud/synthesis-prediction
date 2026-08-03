"""Build connected validation groups enforcing DOI, scaffold and pair isolation.

Records are joined into the same component if they share *any* protected
identity. Splitting components therefore guarantees that no DOI, ligand
structure/scaffold surrogate or metal-linker pair crosses train/test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    data = pd.read_csv(ROOT / "data/v12_training_candidates.csv").fillna("")
    n = len(data)
    uf = UnionFind(n)
    protected = {
        "doi": data["Source_DOI"].astype(str),
        "ligand_identity": data["Ligand_InChIKey"].astype(str).where(
            data["Ligand_InChIKey"].astype(str).ne(""), data["Legante"].astype(str).str.casefold()
        ),
        "metal_ligand_pair": data["Metal_Ligand_Group"].astype(str),
    }
    for values in protected.values():
        first = {}
        for i, value in enumerate(values):
            if not value:
                continue
            if value in first:
                uf.union(i, first[value])
            else:
                first[value] = i

    roots = [uf.find(i) for i in range(n)]
    labels = {root: f"SAFE-GROUP-{j+1:03d}" for j, root in enumerate(sorted(set(roots)))}
    data["Leakage_Safe_Group"] = [labels[root] for root in roots]
    data.to_csv(ROOT / "data/v12_training_candidates_grouped.csv", index=False)

    group_classes = data.groupby("Leakage_Safe_Group")["Outcome_Class"].agg(
        records="size", classes="nunique"
    )
    report = {
        "records": n,
        "leakage_safe_connected_groups": int(data["Leakage_Safe_Group"].nunique()),
        "largest_group_records": int(group_classes["records"].max()),
        "five_fold_validation_viable": bool(data["Leakage_Safe_Group"].nunique() >= 5),
        "all_groups_multiclass": bool(group_classes["classes"].ge(2).all()),
        "policy": "No DOI, ligand identity/scaffold surrogate or metal-linker pair may cross a fold boundary.",
    }
    (ROOT / "reports/leakage_safe_group_readiness_v10_12.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
