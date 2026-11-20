"""Stratified sample for manual audit: 6 per label (seed 7), 2-hop and 3-hop.
Prints sub-question answers and where each intermediate answer string appears in retrieved docs.
Usage: python sample_audit.py out/ > audit_sample.txt"""
import json, sys, random
d = sys.argv[1]
C = json.load(open(f"{d}/corpus.json")); R = [json.loads(l) for l in open(f"{d}/labels.jsonl")]
random.seed(7); S = []
for lab, quota in [("SUFF1", 6), ("RECOV", 6), ("UNREC", 6)]:
    pool = [r for r in R if r["label"] == lab and r["hops"] == 2]
    pool3 = [r for r in R if r["label"] == lab and r["hops"] == 3]
    k3 = min(1, len(pool3)); S += random.sample(pool, min(quota - k3, len(pool))) + random.sample(pool3, k3)
for n, r in enumerate(S, 1):
    print(f"\n#### P{n:02d} {r['id']} LABEL={r['label']} cov1={r['cov1']} cov12={r['cov12']}/{r['n_supp']}")
    print("Q:", r["question"], "| gold:", r["answer"], r["aliases"][:3])
    hops = [(s["question"], s["answer"]) for s in r["decomp"]]
    print("decomp:", hops); print("Q2:", r["q2"])
    for rnd, ids in (("R1", r["r1"]), ("R2", r["r2"])):
        for d_ in ids:
            t = C[d_]["text"]; g = "G" if d_ in r["supp"] else " "
            hits = [a for _, a in hops if a.lower() in t.lower()]
            print(f"  {rnd}[{g}] {C[d_]['title']} | hits={hits} | {t[:230]}")
