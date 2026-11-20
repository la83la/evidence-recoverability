# MuSiQue pilot: manual audit of 18 cases (reviewed by Claude, not a human annotator)
Sample: sample_audit.py seed 7, 6 per label (5x 2-hop + 1x 3-hop).

| label | n | gold-coverage label matches reading of passages | notes |
|---|---|---|---|
| SUFF1 | 6 | 6/6 | P05: the key Compaq-1994 sentence sits in a non-gold paragraph; label still correct |
| RECOV | 6 | 6/6 | P11: round-2 query pointed the wrong way ("...to Britain") but still hit the gold paragraph, so lucky recovery |
| UNREC | 6 | 6/6 on coverage | but the cause varies, see below |

UNREC causes (the important finding):
- P16, P17, P18, P14: **the round-2 query generator failed** (copied a distractor entity, repeated hop 1, or hallucinated an entity). A correct bridge query would very likely recover → these are "unrecoverable under THIS strategy", not by the corpus.
- P13: weak query (latched onto a film title).
- P15: **the question itself is ill-posed** ("the country it is about"; no referent). Only the user can clarify → this is a natural "ask" case.
=> The UNREC label depends heavily on the strength of the round-2 strategy. That is consistent with the definition (fixed strategy), but the paper needs a second, stronger strategy (e.g. oracle-decomposition query or a 7B model) to separate "strategy failure" from "corpus gap".
