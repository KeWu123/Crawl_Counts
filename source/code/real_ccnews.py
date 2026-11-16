"""Measure the size-biased law on a real news crawl (CC-News, 2 shards, 283k articles).

The paper's object is a corpus assembled from *sources* but consumed as *documents*.
This script instantiates that object on real data:

  cleaning   keep prose articles (>=15 sentence boundaries), strip per-outlet
             boilerplate (shingles present in >50% of an outlet's articles)
  clustering MinHash-LSH over 5-word shingles of the lede, every candidate pair
             verified by exact Jaccard; a threshold tau induces a partition
  sources    the reference partition is tau = 0.90; a story republished m times
             is one source with m documents
  measures   TV(P, Q)    the size bias of counting documents (tau-free)
             TV(P, P_w)  the part a detector at threshold tau fails to remove
             where P is the law over stories, one unit per story, and Q one unit
             per document

Writes paper/alt/ccnews.json.  Reproduce: python code/real_ccnews.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ccnews"
OUT = ROOT / "paper" / "alt" / "ccnews.json"

NSHINGLE = 120
K = 5
NHASH = 128
BANDS, ROWS = 32, 4
MIN_PERIODS = 15
BOILER_DF = 0.5
TAU_REF = 0.90
TAUS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

TOK = re.compile(r"[a-z0-9]+")
STOP = {"the", "and", "of", "to", "in", "a", "is", "that", "for", "on"}


def token_ids(text: str, vocab: dict) -> np.ndarray:
    toks = TOK.findall(text[:1600].lower())
    if len(toks) < 120 or len(STOP.intersection(toks)) < 3:
        return np.empty(0, dtype=np.int64)
    ids = np.empty(len(toks), dtype=np.int64)
    for i, t in enumerate(toks):
        v = vocab.get(t)
        if v is None:
            v = len(vocab) + 1
            vocab[t] = v
        ids[i] = v
    return ids


def shingle_hashes(ids: np.ndarray, n: int = NSHINGLE) -> np.ndarray:
    if len(ids) < K:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(len(ids) - K + 1, dtype=np.uint64)
    B = np.uint64(1000003)
    for j in range(K):
        h = h * B + ids[j: len(ids) - K + 1 + j].astype(np.uint64)
    h = np.unique(h)
    if len(h) > n:
        h = np.partition(h, n - 1)[:n]
    return np.sort(h)


def load_corpus(keep_texts=False, keep_meta=False, keep_locations=False):
    """Prose filter, then per-outlet boilerplate removal.

    keep_meta returns the url, title and full date string of every surviving
    document, aligned with the returned shingles; the provenance anchors of
    real_provenance.py are built from them and from nothing else.
    keep_locations appends (shard filename, physical zero-based row) for every
    retained record, after the optional texts, preserving repeated-URL identity.
    """
    raw, doms, dates, texts = [], [], [], []
    urls, titles, days, locations = [], [], [], []
    vocab: dict = {}
    dv: dict = {}
    cols = ["text", "domain", "date"] + (["url", "title"] if keep_meta else [])
    for shard in sorted(DATA.glob("shard*.parquet")):
        pf = pq.ParquetFile(shard)
        row_offset = 0
        for b in pf.iter_batches(batch_size=4000, columns=cols):
            d = b.to_pydict()
            for j, (t, do, dt) in enumerate(zip(d["text"], d["domain"], d["date"])):
                if not t or t.count(". ") < MIN_PERIODS:
                    continue
                hs = shingle_hashes(token_ids(t, vocab), n=200)
                if len(hs) < 40:
                    continue
                if do not in dv:
                    dv[do] = len(dv)
                raw.append(hs)
                if keep_locations:
                    locations.append((shard.name, row_offset + j))
                doms.append(dv[do])
                if keep_texts:
                    texts.append(t)
                if keep_meta:
                    urls.append(d["url"][j] or "")
                    titles.append(d["title"][j] or "")
                    days.append(dt or "")
                dates.append(int(dt[:7].replace("-", "")) if dt else 0)
            row_offset += b.num_rows
    by = defaultdict(list)
    for i, d in enumerate(doms):
        by[d].append(i)
    drop = set()
    for _, idxs in by.items():
        if len(idxs) < 10:
            continue
        df = defaultdict(int)
        for i in idxs:
            for h in raw[i].tolist():
                df[h] += 1
        drop.update(h for h, c in df.items() if c > BOILER_DF * len(idxs))
    flat, keep, dom2 = [], [], []
    for i, hs in enumerate(raw):
        h2 = hs[~np.isin(hs, np.fromiter(drop, np.uint64))] if drop else hs
        if len(h2) < 30:
            continue
        flat.append(h2[:NSHINGLE])
        keep.append(i)
    offs = [0]
    for a in flat:
        offs.append(offs[-1] + len(a))
    out = (np.concatenate(flat), np.asarray(offs),
           np.asarray([doms[i] for i in keep], np.int32),
           np.asarray([dates[i] for i in keep], np.int32), len(dv))
    if keep_meta:
        out = out + ([urls[i] for i in keep], [titles[i] for i in keep],
                     [days[i] for i in keep])
    if keep_texts:
        out = out + ([texts[i] for i in keep],)
    if keep_locations:
        out = out + ([locations[i] for i in keep],)
    return out


def minhash(flat, offs, n_docs):
    rng = np.random.default_rng(2027)
    a = (rng.integers(1, 2**63, size=NHASH, dtype=np.uint64) | np.uint64(1))
    sig = np.empty((n_docs, NHASH), dtype=np.uint64)
    CH = 4000
    for s in range(0, n_docs, CH):
        e = min(s + CH, n_docs)
        rows = [flat[offs[i]:offs[i + 1]] for i in range(s, e)]
        for j in range(NHASH):
            sig[s:e, j] = [(r * a[j]).min() for r in rows]
    return sig


def lsh_pairs(sig, max_group=200):
    n = sig.shape[0]
    pairs = set()
    for b in range(BANDS):
        blk = sig[:, b * ROWS:(b + 1) * ROWS]
        key = np.zeros(n, dtype=np.uint64)
        for j in range(ROWS):
            key = key * np.uint64(1000003) + blk[:, j]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        start = 0
        for i in range(1, n + 1):
            if i == n or ks[i] != ks[start]:
                if i - start > 1:
                    grp = order[start:i]
                    if len(grp) > max_group:      # star, not a cap: no cluster is truncated
                        head, rest = int(grp[0]), grp[1:]
                        for v in rest:
                            v = int(v)
                            pairs.add((head, v) if head < v else (v, head))
                    else:
                        for ii in range(len(grp)):
                            for jj in range(ii + 1, len(grp)):
                                u, v = int(grp[ii]), int(grp[jj])
                                pairs.add((u, v) if u < v else (v, u))
                start = i
    return pairs


def exact_jaccard(flat, offs, pairs):
    out = []
    for u, v in pairs:
        a = set(flat[offs[u]:offs[u + 1]].tolist())
        b = set(flat[offs[v]:offs[v + 1]].tolist())
        inter = len(a & b)
        if inter:
            out.append((u, v, inter / (len(a) + len(b) - inter)))
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 3))


def partition(n, edges, tau):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, s in edges:
        if s >= tau:
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[ru] = rv
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def main():
    flat, offs, domain, dates, n_outlets = load_corpus()
    n = len(offs) - 1
    print(f"usable documents: {n}   outlets: {n_outlets}")
    sig = minhash(flat, offs, n)
    cand = lsh_pairs(sig)
    edges = exact_jaccard(flat, offs, cand)
    print(f"candidate pairs {len(cand)}, verified {len(edges)}")

    ref = partition(n, edges, TAU_REF)
    sizes = np.bincount(ref)
    n_story = len(sizes)
    print(f"stories at tau={TAU_REF}: {n_story}, mean m={n/n_story:.3f}, max m={sizes.max()}")

    srt = np.sort(sizes)[::-1]
    cum = np.cumsum(srt) / srt.sum()
    top1 = max(1, int(round(0.01 * n_story)))

    story_of = ref
    order = np.argsort(story_of, kind="stable")
    bounds = np.searchsorted(story_of[order], np.arange(n_story + 1))
    m_story = np.bincount(story_of, minlength=n_story).astype(float)

    # extra copies that sit on an outlet other than the story's most frequent one
    extra_off_mode = 0
    multi_outlet_stories = 0
    for s in range(n_story):
        idx = order[bounds[s]:bounds[s + 1]]
        if len(idx) < 2:
            continue
        cnt = np.bincount(domain[idx], minlength=n_outlets)
        extra_off_mode += len(idx) - cnt.max()
        if (cnt > 0).sum() > 1:
            multi_outlet_stories += 1

    # TV between the source law and the document law, and between the source law
    # and its reweighted version: both are 1/2 E|.-1| of a ratio, no attribute needed
    tv_ratio = lambda r: float(0.5 * np.abs(r - 1).mean())
    TV_PQ = tv_ratio(m_story / m_story.mean())
    # story-level bootstrap: the tilt is a mean over stories, so resample stories
    rng_b = np.random.default_rng(2027)
    boot = np.empty(400)
    for b_ in range(400):
        idx = rng_b.integers(0, n_story, n_story)
        mb = m_story[idx]
        boot[b_] = 0.5 * np.abs(mb / mb.mean() - 1).mean()
    TV_PQ_SE = float(boot.std(ddof=1))
    m_synd = np.ones(n_story)
    for s_ in range(n_story):
        idx = order[bounds[s_]:bounds[s_ + 1]]
        m_synd[s_] = len(np.unique(domain[idx])) if len(idx) else 1.0
    TV_PQ_synd = tv_ratio(m_synd / m_synd.mean())

    res = {
        "n_docs": int(n), "n_outlets": int(n_outlets),
        "n_stories": int(n_story), "mean_m": float(n / n_story),
        "max_m": int(sizes.max()),
        "share_docs_m_ge2": float(sizes[sizes >= 2].sum() / n),
        "share_docs_m_ge5": float(sizes[sizes >= 5].sum() / n),
        "share_stories_singleton": float((sizes == 1).mean()),
        "top1pct_story_share": float(cum[top1 - 1]),
        "top10pct_story_share": float(cum[max(1, int(round(0.1 * n_story))) - 1]),
        "m_quantiles": {str(q): int(np.quantile(sizes, q, method="higher"))
                        for q in (0.5, 0.9, 0.99)},
        "TV_PQ": TV_PQ, "TV_PQ_boot_se": TV_PQ_SE, "TV_PQ_syndication": TV_PQ_synd,
        "mean_ratio": float((m_story / m_story.mean()).mean()),
        "share_extra_off_mode": float(extra_off_mode / max(1, n - n_story)),
        "multi_outlet_stories": int(multi_outlet_stories),
        "sweep": [],
    }
    print(f"TV(P,Q) = {res['TV_PQ']:.4f}")

    ref_pairs = None
    for tau in TAUS:
        inv = partition(n, edges, tau)
        size = np.bincount(inv)
        w = 1.0 / size[inv]
        U = np.bincount(story_of, weights=w, minlength=n_story)
        up = U / U.mean()
        same = inv[edges[:, 0].astype(int)] == inv[edges[:, 1].astype(int)]
        gold = edges[:, 2] >= TAU_REF
        tp = float((same & gold).sum()); fp = float((same & ~gold).sum())
        fn = float((~same & gold).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        # the tilt this partition itself implies: clusters act as the sources
        m_tau = np.bincount(inv).astype(float)
        tilt_tau = 0.5 * np.abs(m_tau / m_tau.mean() - 1).mean()
        row = {"tau": tau, "n_clusters": int(inv.max() + 1),
               "share_merged": float(1 - (size == 1).mean()),
               "share_docs_merged": float(1 - size[size == 1].sum() / n),
               "over": float(0.5 * np.maximum(1 - up, 0).mean()),
               "under": float(0.5 * np.maximum(up - 1, 0).mean()),
               "pair_f1": f1, "over_merged_pairs": int(fp),
               "TV_PPw": tv_ratio(up), "tilt_own": float(tilt_tau),
               "etatilde": float(np.abs(up - 1).mean()),
               "max_U_ratio": float(up.max())}
        res["sweep"].append(row)
        print(f"  tau={tau:.2f} clusters={row['n_clusters']:6d} merged={row['share_merged']:.3f} "
              f"F1={f1:.3f} TV(P,Pw)={row['TV_PPw']:.4f} eta={row['etatilde']:.4f}")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
