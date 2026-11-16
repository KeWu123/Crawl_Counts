"""Does the tilt of a real corpus damage a decision, or only the number a corpus reports?

Section 5 prices the tilt of the document law but never lets a learner act on it.  This
script closes that loop on the same 50,083 CC-News documents:

  task      predict the section of an article (nine classes) from its first 200 tokens,
            labels taken from the URL path --- content-blind, and fixed before any
            deduplication, so the label is not a product of the partition
  detector  MinHash-LSH at threshold tau gives a partition; the weight of a document is
            1/m of its cluster, which is what keep-one-per-cluster does in expectation
  learner   multinomial logistic regression on hashed 1-2 grams, trained with those
            weights, five seeds
  report    the error the corpus reports: the document-weighted held-out error
  truth     the error under the source law: the same held-out documents reweighted by
            1/m of the title+day codebook, the content-blind partition of §5
  decision  L_P(h_tau) - min_tau' L_P(h_tau'), i.e. what the choice of tau costs the task

Held-out documents are split by source, so no cluster straddles the split.

Writes paper/alt/task.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from real_ccnews import (ROOT, TAUS, TAU_REF, exact_jaccard, load_corpus,
                         lsh_pairs, minhash, partition)
from real_provenance import group_of, norm_title

OUT = ROOT / "paper" / "alt" / "task.json"

SECTIONS = {
    "sport": "sport", "sports": "sport", "sports-story": "sport", "football": "sport",
    "soccer": "sport", "nfl": "sport",
    "world": "world", "international": "world", "global": "world",
    "business": "business", "money": "business", "economy": "business",
    "finance": "business", "markets": "business",
    "entertainment": "ent", "celebrity": "ent", "celebrities": "ent", "movies": "ent",
    "film": "ent", "music": "ent", "tv": "ent", "television": "ent", "arts": "ent",
    "culture": "ent", "books": "ent",
    "politics": "politics", "political": "politics", "election": "politics",
    "opinion": "opinion", "editorial": "opinion",
    "tech": "tech", "technology": "tech",
    "health": "health", "lifestyle": "life", "life": "life", "food": "life",
    "travel": "life", "fashion": "life",
}
MIN_CLASS = 200
MAXTOK = 200
BUCKETS = 1 << 17
SEEDS = (11, 12, 13, 14, 15)
EPOCHS = 24
BATCH = 256
LR = 0.05


def section_of(url: str) -> str:
    segs = [s for s in (url or "").split("/") if s]
    for s in segs[1:4]:
        s2 = s.lower().split("?")[0]
        if s2 in SECTIONS:
            return SECTIONS[s2]
    return ""


def fnv(tok: str) -> int:
    h = 0xCBF29CE484222325
    for ch in tok.encode():
        h ^= ch
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def features(texts, buckets=BUCKETS):
    rows, cols, vals = [], [], []
    for r, t in enumerate(texts):
        toks = [w for w in "".join(c if c.isalnum() else " " for c in t[:1400].lower()).split()
                if len(w) > 1][:MAXTOK]
        seen = set()
        for w in toks:
            j = fnv(w) % buckets
            if j not in seen:
                seen.add(j)
                rows.append(r); cols.append(j); vals.append(1.0)
        for a, b in zip(toks, toks[1:]):
            j = fnv(a + " " + b) % buckets
            if j not in seen:
                seen.add(j)
                rows.append(r); cols.append(j); vals.append(1.0)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), buckets), dtype=np.float32)
    norm = np.sqrt(np.asarray(X.sum(1)).ravel())
    norm[norm == 0] = 1.0
    return sparse.diags(1.0 / norm).dot(X).tocsr()


def train(X, y, w, n_class, seed):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((d, n_class), dtype=np.float32)
    b = np.zeros(n_class, dtype=np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0
    w = w.astype(np.float32)
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]
            logits = Xb @ W + b
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), y[idx]] -= 1.0
            wb = w[idx][:, None] / w[idx].sum()
            G = Xb.T @ (p * wb) + 1e-5 * W
            gb = (p * wb).sum(0)
            t += 1
            for P, G_, M, V in ((W, G, mW, vW), (b, gb, mb, vb)):
                M *= 0.9; M += 0.1 * G_
                V *= 0.999; V += 0.001 * G_ * G_
                P -= LR * (M / (1 - 0.9 ** t)) / (np.sqrt(V / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def main():
    (flat, offs, domain, dates, n_outlets,
     urls, titles, days, texts) = load_corpus(keep_texts=True, keep_meta=True)
    n = len(offs) - 1
    print(f"documents {n}")

    lab = np.array([section_of(u) for u in urls])
    keep = np.array([s != "" for s in lab])
    counts = {s: int((lab == s).sum()) for s in set(lab[keep])}
    keep &= np.array([counts.get(s, 0) >= MIN_CLASS for s in lab])
    idx = np.where(keep)[0]
    print(f"labelled documents {len(idx)} in {len(counts)} sections, "
          f"kept {len({lab[i] for i in idx})}: {sorted({lab[i] for i in idx})}")

    classes = sorted({lab[i] for i in idx})
    cls_of = {c: k for k, c in enumerate(classes)}
    y = np.array([cls_of[lab[i]] for i in idx])
    X = features([texts[i] for i in idx])

    # partitions on the full corpus
    sig = minhash(flat, offs, n)
    edges = exact_jaccard(flat, offs, lsh_pairs(sig))
    src = group_of([norm_title(t) + "|" + d[:10] for t, d in zip(titles, days)])
    m_src = np.bincount(src).astype(float)
    src_held = np.zeros(len(m_src), dtype=bool)
    rng = np.random.default_rng(7)
    hold_src = rng.permutation(len(m_src))
    src_held[hold_src[: int(0.2 * len(m_src))]] = True

    src_i = src[idx]
    m_src_i = m_src[src_i]
    held = src_held[src_i]
    tr, te = ~held, held
    print(f"held-out documents {te.sum()} on {len(set(src_i[te]))} sources")

    # the source law on the held-out documents: one unit per source
    w_true = (1.0 / m_src_i)[te]
    w_true = w_true / w_true.sum()

    def evaluate(W, b):
        logits = X[te] @ W + b
        err = (logits.argmax(1) != y[te]).astype(float)
        rep = err.mean()                      # what the document-weighted split reports
        tru = float((err * w_true).sum())     # what the source law says
        return rep, tru, err

    def bootstrap(err):
        """"Paired bootstrap over held-out sources: the two numbers are means of the
        same per-document errors under two weightings, so only their difference is precise."""
        codes, inv = np.unique(src_i[te], return_inverse=True)
        n_s = np.bincount(inv).astype(float)
        s_err = np.bincount(inv, weights=err)
        rng_b = np.random.default_rng(2027)
        out = np.empty((600, 3))
        for k in range(600):
            pick = rng_b.integers(0, len(codes), len(codes))
            nn = n_s[pick].sum()
            rep = s_err[pick].sum() / nn
            tru = (s_err[pick] / n_s[pick]).mean()
            out[k] = (rep - tru, rep, tru)
        sd = out.std(axis=0, ddof=1)
        return float(sd[0]), float(sd[1]), float(sd[2])

    runs = {}
    settings = [("naive", None)] + [(f"tau={t:.2f}", t) for t in TAUS]
    for name, tau in settings:
        if tau is None:
            m = np.ones(n)
        else:
            part = partition(n, edges, tau)          # cluster id of every document
            m = np.bincount(part)[part].astype(float)
        wtrain = np.where(tr, 1.0 / m[idx], 0.0)
        wtrain[~tr] = 0.0
        errs, reps, trus = [], [], []
        for seed in SEEDS:
            W, b = train(X[tr], y[tr], wtrain[tr], len(classes), seed)
            r_, t_, e_ = evaluate(W, b)
            reps.append(r_); trus.append(t_); errs.append(e_)
        err = np.mean(errs, axis=0)
        reps = np.asarray(reps); trus = np.asarray(trus)
        # exact decomposition: E_Q[err] - E_P[err] = Cov_P(m, err) / kappa.  Per source,
        # err_z is the mean held-out error and m_z the size of that source in the corpus.
        codes_z = np.unique(src_i[te])
        m_z = np.array([m_src[c] for c in codes_z], dtype=float)
        e_z = np.array([err[src_i[te] == c].mean() for c in codes_z])
        kappa = m_z.mean()
        cov = float((m_z * e_z).mean() - kappa * e_z.mean())
        sd_diff, sd_rep, sd_tru = bootstrap(err)
        runs[name] = {"tau": tau, "report": float(reps.mean()), "report_se": float(np.std(reps, ddof=1)),
                      "truth": float(trus.mean()), "truth_se": float(np.std(trus, ddof=1)),
                      "gap_report": float(reps.mean() - trus.mean()),
                      "gap_report_se": sd_diff,
                      "report_cluster_se": sd_rep, "truth_cluster_se": sd_tru,
                      "cov_over_kappa": cov / kappa,
                      "n_sources_held": int(len(codes_z)),
                      "kappa_source": float(kappa),
                      "n_train": int(tr.sum()),
                      "eff_n": float(1.0 / (w_true ** 2).sum()),
                      "majority_accuracy": float((y[te] == np.bincount(y[tr]).argmax()).mean()),
                      "class_support": {str(c): int((y[te] == c).sum()) for c in np.unique(y[te])}}
        print(f"{name:9s} report={reps.mean():.4f} truth={trus.mean():.4f} "
              f"diff={reps.mean()-trus.mean():+.4f}")

    best = min(r["truth"] for r in runs.values())
    for name, r in runs.items():
        r["gap"] = r["truth"] - best
    res = {"n_docs": int(len(idx)), "classes": classes,
           "n_held": int(te.sum()), "seeds": list(SEEDS),
           "runs": runs,
           "max_report_gap": max(abs(r["gap_report"]) for r in runs.values()),
           "max_decision_gap": max(r["gap"] for r in runs.values())}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"max |report - truth| = {res['max_report_gap']:.4f}, "
          f"max decision gap = {res['max_decision_gap']:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
