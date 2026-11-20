"""Decision-time feasibility: can round-1-only signals predict the label?

Visible at decision time: question, question date, round-1 ranked turns + BM25 scores + text,
memory size, and (group 'q2') the follow-up query text the LLM wrote from round 1.
NOT used: round-2 results, gold, special flags.

Tasks
  T1 SUFF1 vs not-SUFF1                 (answerable items)
  T2 RECOV vs UNREC                     (answerable, not SUFF1)        <- recoverability
  T3 RECOV vs NOANS                     (search would help vs info absent -> ask)
  T4 RECOV vs (UNREC ∪ NOANS)           ("search again?" decision)
CV: GroupKFold(5) by `group` (LongMemEval: question incl. its _cfdel twin; LoCoMo: conversation).

Usage: python features_memory.py out/lme [out/locomo ...]
"""
import json, sys, re, numpy as np
from collections import Counter
import Stemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

STOP = set("a an the of in on at to for by with from and or is was were are be been who what which when where whose how did does do that this as it its his her their i my me you your we our".split())
TEMPORAL = re.compile(r"\b(when|how long|how many (days|weeks|months|years)|ago|last|first|before|after|recent|latest|currently|now|still|since)\b", re.I)
stem = Stemmer.Stemmer("english")


def terms(s):
    return {stem.stemWord(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def feats(o):
    sc = np.array(o["r1_top30_scores"], dtype=float)
    sc = np.pad(sc, (0, max(0, 30 - len(sc))), constant_values=0.0)
    q = terms(o["question"]); dt = [terms(t) for t in o["r1_texts"]]
    union = set().union(*dt) if dt else set()
    per = [len(q & t) / max(1, len(q)) for t in dt] or [0.0]
    sess = o["r1_sess_idx"]; mx = max(1, o["max_sess_idx"])
    # NOTE: memory size (n_units) is deliberately NOT a feature: counterfactual deletion shrinks memory,
    # so it leaks the cfdel condition (alone: AUC 0.679 on LongMemEval T3).
    base = [sc[0], sc[0] - sc[1], sc[:5].mean(), sc[4] - sc[9], sc[:30].std() / max(1e-6, sc[:30].mean()), len(q)]
    lex = [len(q & union) / max(1, len(q)), max(per), float(np.mean(per)),
           sum(sum(w in t for t in dt) >= 2 for w in q) / max(1, len(q))]
    mem = [len(set(sess)) / max(1, len(sess)), float(np.mean(sess)) / mx, max(sess) / mx,
           float(bool(TEMPORAL.search(o["question"])))]
    q2 = terms(o["q2"]); nov = q2 - q
    q2f = [len(nov) / max(1, len(q2)), len(nov & union) / max(1, len(nov)) if nov else 0.0, len(q2)]
    return dict(relevance=[sc[0]], score=base, lex=lex, mem=mem, q2=q2f)


GROUPS = {"relevance(top1)": ["relevance"], "score": ["score"], "score+lex": ["score", "lex"],
          "score+lex+mem": ["score", "lex", "mem"], "all(+q2 text)": ["score", "lex", "mem", "q2"]}
TASKS = {"T1 SUFF1 vs rest": (lambda r: r["label"] in ("SUFF1", "RECOV", "UNREC"), lambda r: r["label"] == "SUFF1"),
         "T2 RECOV vs UNREC": (lambda r: r["label"] in ("RECOV", "UNREC"), lambda r: r["label"] == "RECOV"),
         "T3 RECOV vs NOANS": (lambda r: r["label"] in ("RECOV", "NOANS"), lambda r: r["label"] == "RECOV"),
         "T4 RECOV vs UNREC+NOANS": (lambda r: r["label"] in ("RECOV", "UNREC", "NOANS"), lambda r: r["label"] == "RECOV")}


def cv_auc(X, y, g):
    k = min(5, len(set(g)))
    if k < 2 or y.sum() < k or (len(y) - y.sum()) < k:
        return float("nan")
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    p = cross_val_predict(m, X, y, cv=GroupKFold(k), groups=g, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def report(d):
    rows = [json.loads(l) for l in open(f"{d}/labels.jsonl")]
    print(f"\n===== {d}  ({rows[0]['dataset']}, {len(rows)} items)")
    print("labels:", dict(Counter(r["label"] for r in rows)))
    print("same-budget control:", dict(Counter(r["label_samek_control"] for r in rows)))
    by = {}
    for r in rows:
        by.setdefault(r["qtype"], Counter())[r["label"]] += 1
    for k, v in sorted(by.items()):
        print(f"  {k:28s} {dict(v)}")
    rec = [r for r in rows if r["label"] == "RECOV"]
    print(f"RECOV also recovered by same-query deeper retrieval: {sum(r['label_samek_control']=='RECOV' for r in rec)}/{len(rec)}")
    ku = [r for r in rows if r["qtype"] == "knowledge-update" and r["n_gold"] and r["special"] is None]
    if ku:
        stale = sum(r["latest_cov1"] < r["n_latest"] and r["cov1"] > r["latest_cov1"] for r in ku)
        print(f"knowledge-update: round-1 has old evidence but misses latest: {stale}/{len(ku)}")
    F = [feats(r) for r in rows]
    res = {}
    for tn, (sel, yf) in TASKS.items():
        idx = [i for i, r in enumerate(rows) if sel(r)]
        if not idx:
            continue
        y = np.array([yf(rows[i]) for i in idx], dtype=int); g = np.array([rows[i]["group"] for i in idx])
        print(f"\n{tn}: n={len(idx)} pos={y.sum()} groups={len(set(g))}")
        for gn, parts in GROUPS.items():
            X = np.array([sum((F[i][p] for p in parts), []) for i in idx], dtype=float)
            auc = cv_auc(X, y, g); res[f"{tn} | {gn}"] = auc
            print(f"  {gn:18s} AUC={auc:.3f}")
    json.dump(res, open(f"{d}/feature_auc.json", "w"), indent=1)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
