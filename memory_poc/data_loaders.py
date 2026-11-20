"""Load LongMemEval / LoCoMo into one item format.

Item = dict(
  qid, base_qid, group, dataset, qtype, question, question_date, answer,
  units  = [dict(uid, sess, date, role, text)],   # retrieval units = single turns
  gold   = [uid, ...]   # evidence units ('' list when none)
  gold_level = 'turn' | 'session' | 'none'
  special    = None | 'abstain' | 'adversarial' | 'cfdel'
  latest_gold = [uid, ...]  # evidence units in the most recent evidence session (knowledge-update analysis)
)
special:
  abstain     LongMemEval *_abs: the info was never given -> correct action is ask/abstain
  adversarial LoCoMo category 5: a near-match turn exists but attributes the event to someone else
  cfdel       counterfactual deletion (our construction): evidence removed from memory -> ask/abstain
"""
import json, re, copy


def _lme_units(x, drop_sessions=()):
    units = []
    for si, (sid, date, sess) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])):
        if sid in drop_sessions:
            continue
        for ti, t in enumerate(sess):
            units.append(dict(uid=f"{sid}#{ti}", sess=sid, sess_idx=si, date=date, role=t["role"],
                              text=t["content"], has_answer=bool(t.get("has_answer"))))
    return units


def load_longmemeval(path, cf_delete=False):
    data = json.load(open(path))
    items = []
    for x in data:
        abstain = x["question_id"].endswith("_abs")
        units = _lme_units(x)
        gold = [u["uid"] for u in units if u["has_answer"]]
        level = "turn"
        if not gold and not abstain:
            # no turn-level marks: fall back to every turn of the answer sessions (any-of per session is NOT used;
            # see README 'gold_level=session')
            gold = [u["uid"] for u in units if u["sess"] in set(x["answer_session_ids"])]
            level = "session"
        if abstain:
            gold, level = [], "none"
        # latest evidence session by date string ("YYYY/MM/DD (Day) HH:MM" sorts lexicographically)
        ev_dates = [u["date"] for u in units if u["uid"] in set(gold)]
        latest = [u["uid"] for u in units if u["uid"] in set(gold) and u["date"] == max(ev_dates)] if ev_dates else []
        base = dict(qid=x["question_id"], base_qid=x["question_id"].replace("_abs", ""), group=x["question_id"].replace("_abs", ""),
                    dataset="longmemeval_s", qtype=x["question_type"], question=x["question"],
                    question_date=x.get("question_date", ""), answer=str(x["answer"]), units=units,
                    gold=gold, gold_level=level, special="abstain" if abstain else None, latest_gold=latest)
        items.append(base)
        if cf_delete and not abstain:
            drop = set(x["answer_session_ids"])
            items.append(dict(base, qid=x["question_id"] + "_cfdel", units=_lme_units(x, drop), gold=[],
                              gold_level="none", special="cfdel", latest_gold=[]))
    return items


_EV = re.compile(r"D(\d+):(\d+)")


def load_locomo(path, cf_delete=False):
    data = json.load(open(path))
    items = []
    for ci, conv in enumerate(data):
        c = conv["conversation"]
        sess_keys = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        units = []
        for k in sess_keys:
            date = c.get(k + "_date_time", "")
            for t in c[k]:
                txt = t["text"] + (f" [shares image: {t['blip_caption']}]" if t.get("blip_caption") else "")
                units.append(dict(uid=t["dia_id"], sess=k, sess_idx=int(k.split("_")[1]), date=date,
                                  role=t["speaker"], text=txt))
        uids = {u["uid"] for u in units}
        for qi, q in enumerate(conv["qa"]):
            ev = [f"D{a}:{b}" for a, b in _EV.findall(" ".join(map(str, q.get("evidence", []))))]
            ev = [e for e in ev if e in uids]
            adv = q["category"] == 5
            answer = str(q.get("answer", q.get("adversarial_answer", "")))
            ev_sess = [u["sess_idx"] for u in units if u["uid"] in set(ev)]
            latest = [u["uid"] for u in units if u["uid"] in set(ev) and ev_sess and u["sess_idx"] == max(ev_sess)]
            qid = f"locomo{ci}_q{qi}"
            base = dict(qid=qid, base_qid=qid, group=f"conv{ci}", dataset="locomo", qtype=f"cat{q['category']}",
                        question=q["question"], question_date=units[-1]["date"] if units else "", answer=answer,
                        units=units, gold=ev, gold_level="turn" if ev else "none",
                        special="adversarial" if adv else None, latest_gold=latest)
            items.append(base)
            if cf_delete and not adv and ev:
                drop = set(ev)
                items.append(dict(base, qid=qid + "_cfdel", units=[u for u in units if u["uid"] not in drop],
                                  gold=[], gold_level="none", special="cfdel", latest_gold=[]))
    return items


def load(dataset, path, cf_delete=False):
    return {"longmemeval_s": load_longmemeval, "locomo": load_locomo}[dataset](path, cf_delete)
