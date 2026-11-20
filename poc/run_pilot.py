"""Pilot: label MuSiQue-Ans dev questions as SUFF1 / RECOV / UNREC under a fixed
corpus, retriever, round-2 query strategy and budget.

Round 1 : BM25(question) top-K1 over the pooled dev corpus.
Round 2 : Qwen2.5-3B-Instruct (greedy) writes ONE follow-up query from the
          question + round-1 passages; BM25(query) top-K2 *new* passages.
Labels  : gold supporting-paragraph coverage (proxy), checked manually on a sample.
Round-2 output is used ONLY for labels; decision features (features.py) use round 1 only.

Usage: python run_pilot.py --data musique_ans_v1.0_dev.jsonl --out out/ [--n 2417]
"""
import argparse, json, os, time, hashlib
import bm25s, Stemmer

K1, K2 = 5, 5
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # HF commit used for the reference run
PROMPT = (
    "You are helping answer a multi-hop question using a search engine over Wikipedia paragraphs.\n"
    "Question: {q}\n\nPassages retrieved so far:\n{ctx}\n\n"
    "What piece of information is still missing to answer the question? "
    "Write ONE short keyword search query that would find it. "
    "If an intermediate entity is already known from the passages, use its name in the query. "
    "Output only the query, nothing else."
)


def key(p):
    return hashlib.md5((p["title"] + "\n" + p["paragraph_text"]).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=None, help="HF commit (default model: pinned REVISION)")
    ap.add_argument("--device", default="cuda:0", help='"auto" = shard a large model over all visible GPUs')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data)]

    # ---- pooled, de-duplicated corpus (fixed for every question)
    corpus, kid = [], {}
    for r in rows:
        for p in r["paragraphs"]:
            k = key(p)
            if k not in kid:
                kid[k] = len(corpus)
                corpus.append({"title": p["title"], "text": p["paragraph_text"]})
    print("corpus size", len(corpus))
    stem = Stemmer.Stemmer("english")
    toks = bm25s.tokenize([c["title"] + " " + c["text"] for c in corpus], stopwords="en", stemmer=stem)
    bm = bm25s.BM25(); bm.index(toks)

    def search(queries, k):
        qt = bm25s.tokenize(queries, stopwords="en", stemmer=stem, show_progress=False)
        ids, sc = bm.retrieve(qt, k=k, show_progress=False)
        return ids.tolist(), sc.tolist()

    if a.n:
        rows = rows[: a.n]
    qs = [r["question"] for r in rows]
    t0 = time.time()
    r1_ids, r1_sc = search(qs, 30)  # keep top-30 for the deeper-k control / features
    t_r1 = (time.time() - t0) / len(qs)

    # ---- round-2 query generation (LLM, greedy)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    rev = a.revision or (REVISION if a.model == MODEL else None)
    tok = AutoTokenizer.from_pretrained(a.model, revision=rev); tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, dtype=torch.bfloat16, device_map=a.device)
    prompts = []
    for q, ids in zip(qs, r1_ids):
        ctx = "\n".join(f"[{i+1}] {corpus[d]['title']}: {corpus[d]['text']}" for i, d in enumerate(ids[:K1]))
        msg = [{"role": "user", "content": PROMPT.format(q=q, ctx=ctx)}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    gen, t_gen = [], 0.0
    for i in range(0, len(prompts), a.batch):
        enc = tok(prompts[i:i + a.batch], return_tensors="pt", padding=True).to(lm.device)
        t0 = time.time()
        with torch.no_grad():
            o = lm.generate(**enc, max_new_tokens=32, do_sample=False)
        t_gen += time.time() - t0
        gen += [s.strip().split("\n")[0].strip('"') for s in tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
        print(f"gen {i+len(enc.input_ids)}/{len(prompts)}", flush=True)
    t_gen /= len(prompts)
    t0 = time.time()
    r2_ids_all, r2_sc_all = search(gen, K1 + K2 + 10)
    t_r2 = (time.time() - t0) / len(qs)

    out = []
    for r, q, ids, sc, gq, i2, s2 in zip(rows, qs, r1_ids, r1_sc, gen, r2_ids_all, r2_sc_all):
        supp = [kid[key(p)] for p in r["paragraphs"] if p["is_supporting"]]
        r1 = ids[:K1]
        r2 = [d for d in i2 if d not in r1][:K2]
        cov = lambda S: sum(d in S for d in supp)
        c1, c12, c10 = cov(set(r1)), cov(set(r1) | set(r2)), cov(set(ids[:K1 + K2]))
        lab = "SUFF1" if c1 == len(supp) else ("RECOV" if c12 == len(supp) else "UNREC")
        out.append(dict(id=r["id"], hops=int(r["id"][0]), question=q, answer=r["answer"],
                        aliases=r.get("answer_aliases", []), supp=supp, r1=r1, r1_scores=sc[:K1],
                        r1_top30=ids, r1_top30_scores=sc, q2=gq, r2=r2,
                        cov1=c1, cov12=c12, cov_r1k10=c10, n_supp=len(supp), label=lab,
                        label_k10_control=("SUFF1" if c1 == len(supp) else ("RECOV" if c10 == len(supp) else "UNREC")),
                        decomp=r.get("question_decomposition")))
    json.dump(corpus, open(os.path.join(a.out, "corpus.json"), "w"))
    with open(os.path.join(a.out, "labels.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    json.dump(dict(model=a.model, revision=rev, K1=K1, K2=K2, n=len(out), corpus=len(corpus),
                   sec_per_q_bm25_r1=t_r1, sec_per_q_llm_query=t_gen, sec_per_q_bm25_r2=t_r2,
                   gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], batch=a.batch),
              open(os.path.join(a.out, "timing.json"), "w"), indent=1)
    from collections import Counter
    print(Counter(o["label"] for o in out), Counter(o["label_k10_control"] for o in out))


if __name__ == "__main__":
    main()
