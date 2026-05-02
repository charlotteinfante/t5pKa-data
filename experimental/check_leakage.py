"""Check for data leakage between T5PKA training CSV and external test sets.

Test sets live in T5PKA_EXTERNAL_TEST_SETS/<name>/test.source, where each line is
a microstate pair "prot_smiles>>deprot_smiles" (test.target holds the pKa).

Training CSV has columns "prot_smiles" and "deprot_smiles".

Leakage definitions (all role-agnostic — the unordered pair {prot, deprot}):
  exact:        canonical-SMILES of test pair == canonical-SMILES of train pair
  pair_min:     for some train pair, min(sim_A, sim_B) >= threshold
                where (sim_A, sim_B) is the best of the two orientations
  any_side:     for some train molecule (prot OR deprot column),
                Tanimoto to either test molecule >= threshold

Both ECFP4 fingerprints and canonical SMILES are computed with useChirality
True and False — results reported separately.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_TAUT_ENUM = rdMolStandardize.TautomerEnumerator()
_TAUT_CACHE: dict = {}


def _tautomer_canonicalize(mol):
    """Return a tautomer-canonical Mol, cached by canonical SMILES."""
    key = Chem.MolToSmiles(mol)
    cached = _TAUT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        out = _TAUT_ENUM.Canonicalize(mol)
    except Exception:
        out = mol
    _TAUT_CACHE[key] = out
    return out

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
TEST_SETS_DIR = Path(__file__).parent / "T5PKA_EXTERNAL_TEST_SETS"


def discover_test_sets(root: Path):
    """Find every directory under root containing a test.source file."""
    sources = sorted(root.rglob("test.source"))
    return [(p.parent.relative_to(root).as_posix(), p) for p in sources]


def canonical(smiles: str, use_chirality: bool, tautomer_canon: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not use_chirality:
        Chem.RemoveStereochemistry(mol)
    if tautomer_canon:
        mol = _tautomer_canonicalize(mol)
        if not use_chirality:
            Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=use_chirality)


def fingerprint(smiles: str, use_chirality: bool, tautomer_canon: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not use_chirality:
        Chem.RemoveStereochemistry(mol)
    if tautomer_canon:
        mol = _tautomer_canonicalize(mol)
        if not use_chirality:
            Chem.RemoveStereochemistry(mol)
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, useChirality=use_chirality
    )


def load_test_pairs(path: Path):
    pairs = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if ">>" not in line:
                print(f"  WARN {path.name}:{line_no} no '>>' separator", file=sys.stderr)
                continue
            a, b = line.split(">>", 1)
            pairs.append((a.strip(), b.strip()))
    return pairs


def load_training_pairs(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = (row.get("prot_smiles") or "").strip()
            d = (row.get("deprot_smiles") or "").strip()
            if p and d:
                rows.append((p, d))
    return rows


def precompute(pairs, use_chirality, tautomer_canon=False):
    """Return list of dicts with canonical SMILES and FPs for each side."""
    out = []
    for a, b in pairs:
        ca = canonical(a, use_chirality, tautomer_canon)
        cb = canonical(b, use_chirality, tautomer_canon)
        fa = fingerprint(a, use_chirality, tautomer_canon)
        fb = fingerprint(b, use_chirality, tautomer_canon)
        out.append({"raw": (a, b), "canon": (ca, cb), "fp": (fa, fb)})
    return out


def check_leakage(test_pairs, train_pairs, use_chirality, tautomer_canon=False):
    """Return per-test-row leakage info and aggregate counters."""
    test_rec = precompute(test_pairs, use_chirality, tautomer_canon)
    train_rec = precompute(train_pairs, use_chirality, tautomer_canon)

    train_pair_canon_set = set()
    train_mol_canon_set = set()
    for r in train_rec:
        ca, cb = r["canon"]
        if ca and cb:
            train_pair_canon_set.add(frozenset({ca, cb}))
        if ca:
            train_mol_canon_set.add(ca)
        if cb:
            train_mol_canon_set.add(cb)

    train_fps_a = [r["fp"][0] for r in train_rec]
    train_fps_b = [r["fp"][1] for r in train_rec]
    train_fps_all = [fp for fp in train_fps_a + train_fps_b if fp is not None]

    per_row = []
    for tr in test_rec:
        ca, cb = tr["canon"]
        fa, fb = tr["fp"]
        info = {
            "smiles_a": tr["raw"][0],
            "smiles_b": tr["raw"][1],
            "exact_pair": False,
            "exact_any_side": False,
            "best_pair_sim": 0.0,
            "best_any_sim": 0.0,
        }
        if ca and cb and frozenset({ca, cb}) in train_pair_canon_set:
            info["exact_pair"] = True
        if (ca and ca in train_mol_canon_set) or (cb and cb in train_mol_canon_set):
            info["exact_any_side"] = True

        if fa is not None and fb is not None:
            sims_a_to_pa = DataStructs.BulkTanimotoSimilarity(fa, train_fps_a)
            sims_a_to_pb = DataStructs.BulkTanimotoSimilarity(fa, train_fps_b)
            sims_b_to_pa = DataStructs.BulkTanimotoSimilarity(fb, train_fps_a)
            sims_b_to_pb = DataStructs.BulkTanimotoSimilarity(fb, train_fps_b)

            best_pair = 0.0
            for i in range(len(train_rec)):
                # orientation 1: test_a<->train_a, test_b<->train_b
                o1 = min(sims_a_to_pa[i], sims_b_to_pb[i])
                # orientation 2: test_a<->train_b, test_b<->train_a
                o2 = min(sims_a_to_pb[i], sims_b_to_pa[i])
                pair = max(o1, o2)
                if pair > best_pair:
                    best_pair = pair
            info["best_pair_sim"] = best_pair

        # any-side: max sim of either test molecule to ANY training molecule
        best_any = 0.0
        for fp in (fa, fb):
            if fp is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps_all)
            if sims:
                m = max(sims)
                if m > best_any:
                    best_any = m
        info["best_any_sim"] = best_any

        per_row.append(info)

    summary = {
        "exact_pair": sum(1 for r in per_row if r["exact_pair"]),
        "exact_any_side": sum(1 for r in per_row if r["exact_any_side"]),
        "pair_min": {t: sum(1 for r in per_row if r["best_pair_sim"] >= t) for t in THRESHOLDS},
        "any_side": {t: sum(1 for r in per_row if r["best_any_sim"] >= t) for t in THRESHOLDS},
    }
    return per_row, summary


def print_summary(name, n_test, summary, use_chirality, tautomer_canon):
    chir = "chirality=True" if use_chirality else "chirality=False"
    taut = "tautomer=canon" if tautomer_canon else "tautomer=raw"
    print(f"\n  [{chir} | {taut}]  test_pairs={n_test}")
    print(f"    exact pair matches (role-agnostic):       {summary['exact_pair']}")
    print(f"    exact any-side matches (molecule-level):  {summary['exact_any_side']}")
    print(f"    {'threshold':>10}  {'pair_min':>10}  {'any_side':>10}")
    for t in THRESHOLDS:
        print(
            f"    {t:>10.2f}  "
            f"{summary['pair_min'][t]:>10}  "
            f"{summary['any_side'][t]:>10}"
        )


def write_per_row_csv(out_path, name, per_row, use_chirality, tautomer_canon):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "test_set",
                "use_chirality",
                "tautomer_canon",
                "smiles_a",
                "smiles_b",
                "exact_pair_match",
                "exact_any_side_match",
                "best_pair_min_sim",
                "best_any_side_sim",
            ]
        )
        for r in per_row:
            w.writerow(
                [
                    name,
                    use_chirality,
                    tautomer_canon,
                    r["smiles_a"],
                    r["smiles_b"],
                    int(r["exact_pair"]),
                    int(r["exact_any_side"]),
                    f"{r['best_pair_sim']:.4f}",
                    f"{r['best_any_sim']:.4f}",
                ]
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("training_csv", type=Path, help="Path to training CSV")
    ap.add_argument(
        "--per-row-dir",
        type=Path,
        default=None,
        help="Optional directory to write per-row CSV reports",
    )
    ap.add_argument(
        "--test-sets-dir",
        type=Path,
        default=TEST_SETS_DIR,
        help=f"Directory of test sets (default: {TEST_SETS_DIR})",
    )
    ap.add_argument(
        "--tautomer-canon",
        action="store_true",
        help="Also run a pass with rdMolStandardize.TautomerEnumerator().Canonicalize() "
             "applied before canonicalization and fingerprinting. Catches leakage where "
             "test and training pairs use different tautomers of the same chemistry.",
    )
    args = ap.parse_args()

    print(f"Training CSV: {args.training_csv}")
    train_pairs = load_training_pairs(args.training_csv)
    print(f"  loaded {len(train_pairs)} training pairs")

    if args.per_row_dir:
        args.per_row_dir.mkdir(parents=True, exist_ok=True)

    test_sets = discover_test_sets(args.test_sets_dir)
    if not test_sets:
        print(f"No test.source files found under {args.test_sets_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Discovered {len(test_sets)} test set(s) under {args.test_sets_dir}")

    taut_modes = (False, True) if args.tautomer_canon else (False,)

    for name, src in test_sets:
        test_pairs = load_test_pairs(src)
        print(f"\n=== {name} ===")
        print(f"  source: {src}  ({len(test_pairs)} pairs)")

        for use_chir in (True, False):
            for taut in taut_modes:
                per_row, summary = check_leakage(test_pairs, train_pairs, use_chir, taut)
                print_summary(name, len(test_pairs), summary, use_chir, taut)
                if args.per_row_dir:
                    chir_tag = "chirT" if use_chir else "chirF"
                    taut_tag = "tautT" if taut else "tautF"
                    stem = args.training_csv.stem
                    safe_name = name.replace("/", "__")
                    out = args.per_row_dir / f"{stem}__{safe_name}__{chir_tag}_{taut_tag}.csv"
                    write_per_row_csv(out, name, per_row, use_chir, taut)


if __name__ == "__main__":
    main()
