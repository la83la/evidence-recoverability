"""Recoverability labels in long-term memory (LongMemEval_S / LoCoMo).

Round 1 : BM25(question) over THIS item's memory (its own turns), top-K1.
Round 2 : LLM (greedy) writes ONE follow-up query from question + date + round-1 turns;
          BM25(query) top-K2 NEW turns.  Used ONLY for labels.
Labels (answerable items, gold = evidence turns):
  SUFF1  gold ⊆ R1
  RECOV  not SUFF1 and gold ⊆ R1 ∪ R2
  UNREC  otherwise
Special items (correct action = ask / abstain, no coverage label):
  NOANS  LongMemEval *_abs, or counterfactual deletion (_cfdel)
  ADVERS LoCoMo category 5 (misleading near-match)

Usage:
  python run_memory.py --dataset longmemeval_s --data data/longmemeval_s_cleaned.json --out out/lme [--cf_delete]
  python run_memory.py --dataset locomo        --data data/locomo10.json             --out out/locomo [--cf_delete]
"""
import argparse, json, os, time
from collections import Counter
import bm25s, Stemmer
from data_loaders import load

PROMPT = (
    "You are an assistant that can search your past conversations with the user(s).\n"
    "Current date: {date}\nQuestion: {q}\n\nConversation snippets retrieved so far:\n{ctx}\n\n"
    "What piece of information is still missing to answer the question? "
    "Write ONE short keyword search query that would find it in the past conversations. "
    "If a relevant name, item or time is already known from the snippets, use it in the query. "
    "Output only the query, nothing else."
)
SNIP = 300  # chars of each retrieved turn shown to the query writer


def unit_text(u):
    return f"[{u['date']}] {u['role']}: {u['text']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["longmemeval_s", "locomo"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k1", type=int, default=10)
    ap.add_argument("--k2", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default=None, help="HF commit; recorded in run_config.json")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0", help='"auto" = shard a large model over all visible GPUs')
    ap.add_argument("--n", type=int, default=None, help="first N base questions (for smoke tests)")
    ap.add_argument("--cf_delete", action="store_true", help="add counterfactual-deletion twins")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = load(a.dataset, a.data, a.cf_delete)
    if a.n:
        keep = list(dict.fromkeys(it["base_qid"] for it in items))[: a.n]
        items = [it for it in items if it["base_qid"] in set(keep)]
    print(f"{len(items)} items", Counter(it["special"] for it in items))
    stem = Stemmer.Stemmer("english")

    # ---- round 1 (per-item index over that item's memory)
    t0 = time.time()
    idx_cache, r1 = {}, []
    for it in items:
        bm = bm25s.BM25()
        bm.index(bm25s.tokenize([unit_text(u) for u in it["units"]], stopwords="en", stemmer=stem, show_progress=False),
                 show_progress=False)
        idx_cache[it["qid"]] = bm
        k = min(max(30, a.k1 + a.k2), len(it["units"]))
        ids, sc = bm.retrieve(bm25s.tokenize([it["question"]], stopwords="en", stemmer=stem, show_progress=False),
                              k=k, show_progress=False)
        r1.append((ids[0].tolist(), sc[0].tolist()))
    t_r1 = (time.time() - t0) / len(items)

    # ---- round-2 query generation
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision); tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16, device_map=a.device)
    prompts = []
    for it, (ids, _) in zip(items, r1):
        ctx = "\n".join(f"[{i+1}] {unit_text(it['units'][d])[:SNIP]}" for i, d in enumerate(ids[: a.k1]))
        msg = [{"role": "user", "content": PROMPT.format(date=it["question_date"], q=it["question"], ctx=ctx)}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    gen, t_gen = [], 0.0
    for i in range(0, len(prompts), a.batch):
        enc = tok(prompts[i:i + a.batch], return_tensors="pt", padding=True).to(lm.device)
        t0 = time.time()
        with torch.no_grad():
            o = lm.generate(**enc, max_new_tokens=32, do_sample=False)
        t_gen += time.time() - t0
        gen += [s.strip().split("\n")[0].strip('"') for s in
                tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
        if (i // a.batch) % 20 == 0:
            print(f"gen {i + len(enc.input_ids)}/{len(prompts)}", flush=True)
    t_gen /= len(prompts)

    # ---- round 2 + labels
    t0 = time.time()
    out = []
    for it, (ids, sc), q2 in zip(items, r1, gen):
        bm = idx_cache[it["qid"]]
        k = min(a.k1 + a.k2 + 10, len(it["units"]))
        i2, _ = bm.retrieve(bm25s.tokenize([q2], stopwords="en", stemmer=stem, show_progress=False), k=k, show_progress=False)
        R1 = ids[: a.k1]
        R2 = [d for d in i2[0].tolist() if d not in R1][: a.k2]
        uid = lambda d: it["units"][d]["uid"]
        g = set(it["gold"])
        s1, s12, s10 = {uid(d) for d in R1}, {uid(d) for d in R1 + R2}, {uid(d) for d in ids[: a.k1 + a.k2]}
        if it["special"] in ("abstain", "cfdel"):
            lab = lab_ctrl = "NOANS"
        elif it["special"] == "adversarial":
            lab = lab_ctrl = "ADVERS"
        elif not g:
            lab = lab_ctrl = "NOGOLD"
        else:
            lab = "SUFF1" if g <= s1 else ("RECOV" if g <= s12 else "UNREC")
            lab_ctrl = "SUFF1" if g <= s1 else ("RECOV" if g <= s10 else "UNREC")
        lg = set(it["latest_gold"])
        out.append(dict(
            qid=it["qid"], base_qid=it["base_qid"], group=it["group"], dataset=it["dataset"], qtype=it["qtype"],
            special=it["special"], question=it["question"], question_date=it["question_date"], answer=it["answer"],
            n_units=len(it["units"]), n_sessions=len({u["sess"] for u in it["units"]}),
            gold=sorted(g), gold_level=it["gold_level"],
            r1=[uid(d) for d in R1], r1_scores=sc[: a.k1], r1_top30_scores=sc,
            r1_texts=[unit_text(it["units"][d])[:600] for d in R1],
            r1_sess_idx=[it["units"][d]["sess_idx"] for d in R1],
            max_sess_idx=max(u["sess_idx"] for u in it["units"]),
            q2=q2, r2=[uid(d) for d in R2], r2_texts=[unit_text(it["units"][d])[:600] for d in R2],
            cov1=len(g & s1), cov12=len(g & s12), n_gold=len(g),
            latest_cov1=len(lg & s1), latest_cov12=len(lg & s12), n_latest=len(lg),
            label=lab, label_samek_control=lab_ctrl))
    t_r2 = (time.time() - t0) / len(items)

    with open(os.path.join(a.out, "labels.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    json.dump(dict(vars(a), n_items=len(out), prompt=PROMPT, snip_chars=SNIP,
                   sec_per_item_bm25_index_and_r1=t_r1, sec_per_item_llm_query=t_gen, sec_per_item_r2=t_r2,
                   gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),
              open(os.path.join(a.out, "run_config.json"), "w"), indent=1, ensure_ascii=False)
    print("labels:", Counter(o["label"] for o in out))
    print("same-budget k1+k2 control:", Counter(o["label_samek_control"] for o in out))


if __name__ == "__main__":
    main()
