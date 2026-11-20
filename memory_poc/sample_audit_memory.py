"""Stratified audit sample: up to 4 items per label (seed 7), printing round-1/round-2 turns with gold marks.
Usage: python sample_audit_memory.py out/longmemeval_s > audit_sample.txt"""
import json, sys, random
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/labels.jsonl")]
random.seed(7)
for lab in ["SUFF1", "RECOV", "UNREC", "NOANS", "ADVERS"]:
    pool = [r for r in rows if r["label"] == lab]
    for r in random.sample(pool, min(4, len(pool))):
        g = set(r["gold"])
        print(f"\n#### {r['qid']} [{r['qtype']}] LABEL={lab} special={r['special']} cov1={r['cov1']} cov12={r['cov12']}/{r['n_gold']}")
        print("Q:", r["question"], "| date:", r["question_date"], "| A:", r["answer"][:200])
        print("Q2:", r["q2"])
        for rnd, ids, txt in (("R1", r["r1"], r["r1_texts"]), ("R2", r["r2"], r["r2_texts"])):
            for u, t in zip(ids, txt):
                print(f"  {rnd}[{'G' if u in g else ' '}] {u} | {t[:220]}")
