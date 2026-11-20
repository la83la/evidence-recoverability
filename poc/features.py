"""Decision-time feasibility check: can round-1-only signals separate
SUFF1 / RECOV / UNREC?  Uses labels.jsonl from run_pilot.py.

Features use ONLY: question, round-1 ranked list + BM25 scores, round-1 passage text,
and (optionally, group 'q2') the follow-up query text the LLM wrote from round 1
(the query is visible before round 2 is executed; its retrieval results are NOT used).

Usage: python features.py out/
"""
import json, sys, re, numpy as np
from collections import Counter
import Stemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

STOP = set("a an the of in on at to for by with from and or is was were are be been who what which when where whose how did does do that this as it its his her their".split())
stem = Stemmer.Stemmer("english")


def terms(s):
    return {stem.stemWord(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def feats(o, corpus):
    sc = np.array(o["r1_top30_scores"], dtype=float)
    q = terms(o["question"])
    docs = [corpus[d] for d in o["r1"]]
    dterms = [terms(d["title"] + " " + d["text"]) for d in docs]
    union = set().union(*dterms)
    cov_union = len(q & union) / max(1, len(q))
    per_doc = [len(q & t) / max(1, len(q)) for t in dterms]
    # how many question terms are covered by >=2 different passages (multi-hop signal)
    multi = sum(sum(w in t for t in dterms) >= 2 for w in q) / max(1, len(q))
    titles_in_q = sum(terms(d["title"]) <= q for d in docs)
    q2 = terms(o["q2"])
    q2_novel = len(q2 - q) / max(1, len(q2))
    q2_from_r1 = len((q2 - q) & union) / max(1, len(q2 - q)) if q2 - q else 0.0
    base = dict(top1=sc[0], gap12=sc[0] - sc[1], mean5=sc[:5].mean(), r5_10=sc[4] - sc[9],
                nqc=sc[:30].std() / max(1e-6, sc[:30].mean()), qlen=len(q))
    lex = dict(cov_union=cov_union, cov_max=max(per_doc), cov_min=min(per_doc), multi=multi,
               titles_in_q=titles_in_q)
    q2f = dict(q2_novel=q2_novel, q2_from_r1=q2_from_r1, q2_len=len(q2))
    return base, lex, q2f


def cv_auc(X, y, seed=0):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    d = sys.argv[1]
    corpus = json.load(open(f"{d}/corpus.json"))
    rows = [json.loads(l) for l in open(f"{d}/labels.jsonl")]
    print("labels:", Counter(r["label"] for r in rows))
    print("k=10 same-query control:", Counter(r["label_k10_control"] for r in rows))
    for h in (2, 3, 4):
        print(f"  {h}-hop:", Counter(r["label"] for r in rows if r["hops"] == h))
    # how much of RECOV is also recovered just by going deeper with the same query?
    rec = [r for r in rows if r["label"] == "RECOV"]
    print(f"RECOV also recovered by same-query k=10: {sum(r['label_k10_control']=='RECOV' for r in rec)}/{len(rec)}")
    F = [feats(r, corpus) for r in rows]
    groups = {"relevance (top1 only)": lambda b, l, q: [b["top1"]],
              "score stats": lambda b, l, q: list(b.values()),
              "score+lexical": lambda b, l, q: list(b.values()) + list(l.values()),
              "score+lexical+q2text": lambda b, l, q: list(b.values()) + list(l.values()) + list(q.values())}
    tasks = {"SUFF1 vs not": ([True] * len(rows), lambda r: r["label"] == "SUFF1"),
             "RECOV vs UNREC (|not SUFF1)": ([r["label"] != "SUFF1" for r in rows], lambda r: r["label"] == "RECOV")}
    res = {}
    for tname, (mask, yf) in tasks.items():
        idx = [i for i, m in enumerate(mask) if m]
        y = np.array([yf(rows[i]) for i in idx], dtype=int)
        print(f"\n{tname}: n={len(idx)} pos={y.sum()}")
        for gname, g in groups.items():
            X = np.array([g(*F[i]) for i in idx])
            aucs = [cv_auc(X, y, s) for s in range(3)]
            res[(tname, gname)] = float(np.mean(aucs))
            print(f"  {gname:28s} AUC={np.mean(aucs):.3f} (±{np.std(aucs):.3f} over CV seeds)")
    # hop-controlled check: does the signal survive within a single hop count?
    print("\nRECOV vs UNREC within 2-hop only (score+lexical+q2text):")
    idx = [i for i, r in enumerate(rows) if r["label"] != "SUFF1" and r["hops"] == 2]
    y = np.array([rows[i]["label"] == "RECOV" for i in idx], dtype=int)
    X = np.array([list(F[i][0].values()) + list(F[i][1].values()) + list(F[i][2].values()) for i in idx])
    print(f"  n={len(idx)} pos={y.sum()} AUC={cv_auc(X, y):.3f}")
    json.dump({f"{k[0]} | {k[1]}": v for k, v in res.items()}, open(f"{d}/feature_auc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
