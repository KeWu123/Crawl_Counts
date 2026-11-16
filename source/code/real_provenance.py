"""Content-blind anchors for the CC-News partition.

real_ccnews.py defines a source by MinHash-LSH over content, then measures the
tilt of that definition.  That number is only as good as the partition, and the
partition is the estimator under indictment.  This script builds three
partitions that never read a shingle -- canonical URL, exact title plus day,
exact title -- and asks three questions:

  1. does the LSH partition look like them at the level of individual pairs,
     over the union of the pairs either partition asserts?
  2. what tilt does each anchor imply, i.e. how wide is the band of defensible
     answers to "how much does counting documents tilt the source law"?
  3. if an anchor is taken as the source law, what residual does a detector at
     threshold tau leave, and does the ordering of the sweep survive?

Writes paper/alt/provenance.json.  Reproduce: python code/real_provenance.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from real_ccnews import (ROOT, TAUS, TAU_REF, exact_jaccard, load_corpus,
                         lsh_pairs, minhash, partition)

OUT = ROOT / "paper" / "alt" / "provenance.json"
NONWORD = re.compile(r"[^a-z0-9 ]+")


def canon_url(u: str) -> str:
    u = (u or "").strip().lower()
    u = u.split("://", 1)[-1]
    u = u.split("#", 1)[0].split("?", 1)[0]
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


def norm_title(t: str) -> str:
    t = NONWORD.sub(" ", (t or "").lower())
    return " ".join(t.split())


def group_of(keys):
    """Dense ids over the distinct keys, first occurrence order."""
    ids = {}
    out = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        v = ids.get(k)
        if v is None:
            v = len(ids)
            ids[k] = v
        out[i] = v
    return out


def tilt(sizes: np.ndarray) -> float:
    s = sizes.astype(float)
    return float(0.5 * np.abs(s / s.mean() - 1).mean())


def tilt_se(sizes: np.ndarray, n_boot: int = 400) -> float:
    rng = np.random.default_rng(2027)
    m = sizes.astype(float)
    out = np.empty(n_boot)
    for b in range(n_boot):
        mb = m[rng.integers(0, len(m), len(m))]
        out[b] = 0.5 * np.abs(mb / mb.mean() - 1).mean()
    return float(out.std(ddof=1))


def pairs_of(part: np.ndarray, cap: int = 300):
    by = {}
    for i, g in enumerate(part.tolist()):
        by.setdefault(g, []).append(i)
    out = set()
    truncated = 0
    for idx in by.values():
        if len(idx) > cap:
            truncated += 1
            idx = idx[:cap]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                out.add((idx[a], idx[b]))
    return out, truncated


def prf(ref_pairs: set, alt_pairs: set):
    tp = len(ref_pairs & alt_pairs)
    p = tp / len(alt_pairs) if alt_pairs else 0.0
    r = tp / len(ref_pairs) if ref_pairs else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f,
            "n_ref_pairs": len(ref_pairs), "n_alt_pairs": len(alt_pairs),
            "n_union": len(ref_pairs | alt_pairs)}


def residual_against(anchor: np.ndarray, n_anchor: int, part: np.ndarray) -> float:
    """TV(P_anchor, target induced by `part`), by Definition 2.1."""
    size = np.bincount(part).astype(float)
    w = 1.0 / size[part]                       # u = 1/m of the detector's cluster
    U = np.bincount(anchor, weights=w, minlength=n_anchor)
    t = U / U.mean()
    return float(0.5 * np.abs(t - 1).mean())


def main():
    (flat, offs, domain, dates, n_outlets,
     urls, titles, days) = load_corpus(keep_meta=True)
    n = len(offs) - 1
    print(f"documents {n}  outlets {n_outlets}")

    sig = minhash(flat, offs, n)
    edges = exact_jaccard(flat, offs, lsh_pairs(sig))
    ref = partition(n, edges, TAU_REF)
    ref_sizes = np.bincount(ref)
    print(f"LSH tau={TAU_REF}: {len(ref_sizes)} clusters, mean m {n/len(ref_sizes):.3f}")

    anchors = {
        "url": group_of([canon_url(u) for u in urls]),
        "title_date": group_of([norm_title(t) + "|" + d[:10] for t, d in zip(titles, days)]),
        "title": group_of([norm_title(t) for t in titles]),
    }

    ref_pairs, trunc = pairs_of(ref)
    res = {"n_docs": int(n), "n_outlets": int(n_outlets),
           "ref_tau": TAU_REF, "ref_clusters": int(len(ref_sizes)),
           "ref_tilt": tilt(ref_sizes), "ref_tilt_se": tilt_se(ref_sizes),
           "ref_pairs": len(ref_pairs), "truncated_groups": trunc,
           "anchors": {}, "sweep": []}

    for name, a in anchors.items():
        sizes = np.bincount(a)
        ap, _ = pairs_of(a)
        r = {"n_groups": int(len(sizes)), "mean_m": float(n / len(sizes)),
             "tilt": tilt(sizes), "tilt_se": tilt_se(sizes),
             "max_m": int(sizes.max())}
        r.update(prf(ref_pairs, ap))
        r["residual_vs_ref_tau"] = residual_against(ref, len(ref_sizes), a)
        res["anchors"][name] = r
        print(f"{name:11s} groups={r['n_groups']:6d} m={r['mean_m']:.3f} "
              f"TV={r['tilt']:.4f}+-{r['tilt_se']:.4f} F1={r['f1']:.3f} "
              f"(P={r['precision']:.3f} R={r['recall']:.3f})")

    # does the tau sweep keep its order when the source law is not the LSH?
    for tau in TAUS:
        inv = partition(n, edges, tau)
        row = {"tau": tau,
               "clusters": int(len(np.bincount(inv))),
               "share_docs_merged": float(1 - np.bincount(inv)[np.bincount(inv) == 1].sum() / n),
               "residual_vs_ref": residual_against(ref, len(ref_sizes), inv)}
        for name, a in anchors.items():
            row[f"residual_vs_{name}"] = residual_against(a, len(np.bincount(a)), inv)
        res["sweep"].append(row)
        print(f"  tau={tau:.2f} merged={row['share_docs_merged']:.3f} "
              f"res_ref={row['residual_vs_ref']:.4f} "
              f"res_url={row['residual_vs_url']:.4f} "
              f"res_td={row['residual_vs_title_date']:.4f} "
              f"res_title={row['residual_vs_title']:.4f}")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
