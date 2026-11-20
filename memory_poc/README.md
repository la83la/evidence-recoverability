# 長期記憶情境的 Recoverability Pilot（LongMemEval_S + LoCoMo）

**研究問題**：在 companion / 長期記憶情境下，第一輪從記憶檢索完之後，只用當下看得到的資訊，能否判斷：
- 證據已經夠了（作答）
- 再搜一輪能補足（再搜）
- 再搜也補不回來，因為使用者沒說過或記憶裡沒有（追問或暫緩作答）

> 這是 **pilot（stage 1：建標籤 + 可分性檢查）**。Stage 2–4 的規格寫在本文第 8 節，**還沒實作**。

---

## 0. 一鍵重跑

```bash
git clone <your-repo> && cd <repo>
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r memory_poc/requirements.txt     # torch 請配合該機器的 CUDA

bash memory_poc/run_all.sh out_mem_smoke 40    # smoke test，約 3 分鐘（含下載約 6GB 模型和 280MB 資料）
bash memory_poc/run_all.sh out_mem             # 全量，RTX 3080 約 15–20 分鐘
diff <(grep -o '"label": "[A-Z0-9]*"' out_mem/longmemeval_s/labels.jsonl | sort | uniq -c) \
     <(grep -o '"label": "[A-Z0-9]*"' memory_poc/reference_results/longmemeval_s/labels.jsonl | sort | uniq -c)
```

可調的環境變數：`DEVICE=cuda:0`、`BATCH=8`、`K1=10`、`K2=10`、`MODEL=...`（換成非預設模型時，不會帶入 revision）、`PY=python`。

**硬體**：
- 1 張 GPU，3B 模型 bf16 約需 7GB VRAM。參考跑用 RTX 3080 10GB、batch 8。
- 7B 模型需要 ≥ 20GB VRAM（`MODEL=Qwen/Qwen2.5-7B-Instruct`）。

---

## 1. 資料

| 資料集 | 來源 | SHA-256 | 規模 |
|---|---|---|---|
| LongMemEval_S（cleaned） | `huggingface.co/datasets/xiaowu0162/longmemeval-cleaned` → `longmemeval_s_cleaned.json` | `d6f21ea9…a3c442` | 500 題；每題約 53 個 session、約 550 個 turn、約 115k tokens |
| LoCoMo | `github.com/snap-research/locomo` → `data/locomo10.json` | `79fa87e9…98ff4` | 10 段對話、1986 題；每段約 19–35 個 session |

`run_all.sh` 會自動下載資料並檢查 SHA-256。

### 統一格式（`data_loaders.py`）

**檢索單位**：單一 turn，索引文字為 `[session 日期] role: content`。

**記憶範圍**：每題只檢索它自己的記憶。LongMemEval 是該題的 haystack，LoCoMo 是該題所屬的那段對話。

**Gold 證據**：

| 資料集 | 一般情況 | 例外 |
|---|---|---|
| LongMemEval | 有 `has_answer=true` 的 turn | 21 題完全沒有 turn 標記，改用 answer session 內**所有** turn（`gold_level=session`，比較嚴格） |
| LoCoMo | `evidence` 欄位的 `D{session}:{turn}` | 無法對應到 turn 的 id 會被丟掉；全部丟光的題目標成 `NOGOLD`，排除在分析外 |

**特殊題（正確行動是追問或暫緩，不算 coverage 標籤）**：

| 標記 | 來源 | 數量 |
|---|---|---|
| `abstain` | LongMemEval `*_abs`：使用者從沒提過 | 30 |
| `cfdel` | **反事實刪除（我們自己構造的）**。LongMemEval：從記憶刪掉該題的 answer session；LoCoMo：刪掉 evidence turn。同一題因此有「原版 / 刪除版」配對 | LongMemEval 470；LoCoMo 見結果 |
| `adversarial` | LoCoMo category 5：記憶裡有相近的 turn，但事件是另一個人的（誤導性近似）| 446 |

---

## 2. 檢索流程

- **Retriever**：BM25（`bm25s`，lucene 變體，`k1=1.5`、`b=0.75`，英文 stopwords，PyStemmer english）。每題各自建一個索引。
- **Round 1**：query = 原始問題，取 top-`K1`（預設 10）。另外保留 top-30 的分數供特徵使用。
- **Round 2（只用來建立標籤）**：
  - Query 生成：`Qwen/Qwen2.5-3B-Instruct`，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，bf16，greedy，`max_new_tokens=32`，取輸出的第一行。
  - 檢索：用生成的 query 取 BM25 結果，排除 round 1 已取過的，取前 `K2`（預設 10）個新 turn。
- **Prompt**（每個 turn 只截前 300 字元）：
  ```
  You are an assistant that can search your past conversations with the user(s).
  Current date: {question_date}
  Question: {q}

  Conversation snippets retrieved so far:
  [1] [date] role: text
  ...

  What piece of information is still missing to answer the question? Write ONE short keyword search query that would find it in the past conversations. If a relevant name, item or time is already known from the snippets, use it in the query. Output only the query, nothing else.
  ```
- **預算**：每題 1 次 LLM 呼叫、2 次 BM25 檢索，最多看 20 個 turn。

---

## 3. 標籤

設 `G` 為 gold turn 集合，`R1` 為第一輪結果，`R2` 為第二輪新取得的結果。

| 標籤 | 條件 | 理想行動 |
|---|---|---|
| `SUFF1` | `G ⊆ R1` | 作答 |
| `RECOV` | 不是 SUFF1，且 `G ⊆ R1 ∪ R2` | 再搜 |
| `UNREC` | 其餘的可回答題 | 換策略、加大預算，或暫緩 |
| `NOANS` | `abstain` 或 `cfdel` | 追問或暫緩 |
| `ADVERS` | LoCoMo cat5 | 拒絕誤導性答案 |

**對照組 `label_samek_control`**：不改寫 query，直接用原始問題取 top-(K1+K2)，文件預算相同。用來區分「改寫 query 的貢獻」和「只是多取幾篇」。

**knowledge-update 分析**：另外記錄 `latest_cov*`，也就是最新那個 evidence session 有沒有被取到，用來找「第一輪只找到舊值」的案例。

---

## 4. 決策時的特徵（`features_memory.py`）

**可以用的**：問題、問題日期、round-1 的排名、分數和文字、記憶大小、LLM 寫的 query 文字。

**不可以用的**：round-2 的結果、gold、`special` 標記。

| 群組 | 特徵 |
|---|---|
| relevance | top-1 BM25 分數（= brief 裡的 relevance-threshold baseline） |
| score | top1、第 1 名減第 2 名、top-5 平均、第 5 名減第 10 名、NQC（top-30 的 std/mean）、問題內容詞數、log(記憶 turn 數) |
| lex | top-K1 合併後的問題詞覆蓋率、單一 turn 的最大覆蓋、平均覆蓋、被 ≥2 個 turn 覆蓋的問題詞比例 |
| mem | R1 來自不同 session 的比例、R1 的平均和最大 session 位置（相對於最新）、問題是否含時間詞 |
| q2 | 新 query 中的新詞比例、這些新詞出現在 R1 的比例、query 長度 |

**分類器**：`StandardScaler` + `LogisticRegression(C=1)`，**GroupKFold(5)**。LongMemEval 以題目分組，原題和它的 `_cfdel` 一定落在同一組；LoCoMo 以對話分組。

| 任務 | 對應的決策 |
|---|---|
| T1 SUFF1 vs. 其他 | 要不要直接作答 |
| T2 RECOV vs. UNREC | 不夠的時候，再搜有沒有用 |
| T3 RECOV vs. NOANS | 該再搜，還是該追問 |
| T4 RECOV vs. UNREC ∪ NOANS | 「要不要再搜一輪」的整體決策 |

---

## 5. 參考結果（本機，RTX 3080，`reference_results/`）

### 標籤分布

| | SUFF1 | RECOV | UNREC | NOANS | ADVERS | NOGOLD |
|---|---|---|---|---|---|---|
| LongMemEval_S（970 筆，含 470 筆 cfdel） | 330 | 42 | 98 | 500 | — | — |
| LoCoMo（3521 筆，含 1535 筆 cfdel） | 797 | 117 | 621 | 1535 | 446 | 5 |

- **RECOV 很少**：LongMemEval 的可回答題中只有 9%，LoCoMo 只有 8%。
- **改寫 query 的貢獻**：RECOV 中，用原始問題取深一點（top-20）也能補足的，LongMemEval 有 25/42，LoCoMo 有 84/117。所以大約 2/3 的「補足」其實只是多取了幾篇。
- **knowledge-update**：第一輪只找到舊值、漏掉最新值的有 6/72 題。
- LongMemEval 的 RECOV 和 UNREC 集中在 multi-session 題（13 / 49）和 temporal 題（13 / 31）。

### 只用第一輪資訊的預測力（AUC，GroupKFold；已移除記憶大小特徵）

| 任務 | LongMemEval：relevance | LongMemEval：全部特徵 | LoCoMo：relevance | LoCoMo：全部特徵 |
|---|---|---|---|---|
| T1 SUFF1 vs. 其他 | 0.798 | 0.817 | 0.670 | 0.739 |
| **T2 RECOV vs. UNREC** | 0.549 | **0.568** | 0.552 | **0.496** |
| T3 RECOV vs. NOANS | 0.680 | 0.858 | 0.529 | 0.554 |
| T4 RECOV vs. UNREC ∪ NOANS | 0.665 | 0.770 | 0.535 | 0.543 |

**解讀**：
- 在記憶情境下，用這些淺層特徵**分不出 RECOV 和 UNREC**（T2 約 0.5–0.57）。這和 MuSiQue 的 0.775 不同。
- T3 在 LongMemEval 有 0.858，在 LoCoMo 只有 0.554。原因可能是 LongMemEval 的 cfdel 刪掉整個 session，相關內容全部消失，所以分數明顯變低；LoCoMo 只刪掉 evidence turn，附近的相關 turn 還留著。**T3 的高分可能有一部分是構造方式造成的**。
- **已排除的洩漏**：記憶大小（`log n_units`）單獨就有 AUC 0.679，因為 cfdel 會讓記憶變小，所以已從特徵中移除。移除後 T3 仍是 0.858。
- 只用 30 題真正的 abstain 當負例時，RECOV vs. ABSTAIN 的 AUC 是 0.728（n=72，樣本很小）。

### 延遲（每筆平攤，RTX 3080，batch 8）

| 步驟 | LongMemEval | LoCoMo |
|---|---|---|
| BM25 建索引 + R1 | 0.057 s | 0.015 s |
| LLM 生成 query | 0.18 s | 0.15 s |
| R2 | 0.0003 s | 0.0002 s |

### cfdel 洩漏檢查（LongMemEval）
答案字串 ≤ 40 字元的 354 題中，刪除版仍在 R1∪R2 找到答案字串的有 15 題，原版有 147 題。這 15 題有一部分可能只是數字巧合，需要人工核對。

---

## 6. 輸出

`OUT/{dataset}/labels.jsonl`：每行一筆。

| 欄位 | 說明 |
|---|---|
| `qid`、`base_qid`、`group`、`qtype`、`special` | 識別與分類 |
| `question`、`question_date`、`answer` | 原始資料 |
| `gold`、`gold_level` | gold 證據 |
| `r1`、`r1_scores`、`r1_top30_scores`、`r1_texts`、`r1_sess_idx` | 第一輪結果 |
| `q2`、`r2`、`r2_texts` | 第二輪 query 和結果 |
| `cov1`、`cov12`、`n_gold`、`latest_cov1`、`latest_cov12`、`n_latest` | 覆蓋數 |
| `label`、`label_samek_control` | 標籤 |

其他輸出：`run_config.json`（所有參數、prompt、延遲）、`run.log`、`features.log`、`feature_auc.json`、`audit_sample.txt`（每類 4 題）。

---

## 7. 重現性與已知限制

- BM25 和特徵是決定性的。LLM query 在同 GPU、同 batch 下可以重現；換 GPU 或 batch size 可能讓少數題目的 q2 改變。
- 標籤是 **proxy**：看 gold turn 有沒有被取到，沒有讓 generator 實際作答。
- `gold_level=session` 的 21 題要求整個 session 都被取到，標籤偏向 UNREC。
- LoCoMo 只有 10 段對話，GroupKFold 實際上只有 10 個獨立單位，AUC 的變異會很大。
- **cfdel 的假設**：刪掉 evidence 之後，記憶裡就沒有答案。洩漏檢查顯示少數例外，需要人工核對。
- LongMemEval 的 abstention 只有 30 題，所以 NOANS 主要來自 cfdel，也就是我們自己構造的資料。論文中必須寫明這一點。

---

## 8. 後續階段（規格，尚未實作）

**Stage 2：作答與 baseline（同一批 R1）**

1. Generator：Qwen2.5-7B-Instruct，或任何可替換的模型。在 R1 下作答，也可以選擇拒答。
2. 用 LLM judge 或字串比對判斷答案對錯，算出「無證據作答率」。
3. Baseline：
   - (a) relevance threshold
   - (b) Sufficient Context 官方 autorater prompt（`github.com/hljoren/sufficientcontext`）
   - (c) 永遠再搜一輪
   - (d) 永遠不搜
   - (e) 本研究的 predictor
   - 可以再加 Stop-RAG 的 value head（`github.com/chosolbee/Stop-RAG`）
4. 切分：LongMemEval 以題目分組切 60/20/20，LoCoMo 留一段對話。Threshold 在 validation 調，在 test 報告。

**Stage 3：成本**

- 報告每題的端到端延遲：決策模組 + 檢索 + 生成。
- 報告檢索次數、token 數、Over-Searching 的 TPC，以及過早停止率。過早停止率 = RECOV 卻被判成不搜的比例。
- 主要比較方式：在相同的無證據作答率下比較上述數字。
- 寬鬆預算：K2=30，或做兩輪第二輪檢索。也可以改用 LongMemEval_M（約 500 個 session）。

**Stage 4：錯誤分析與穩健性**

- 錯誤類型：knowledge-update（只找到舊值）、adversarial（誤導性近似）、cfdel（使用者沒說過）、multi-session（證據分散）。
- 換一種第二輪策略（7B 模型，或 dense retriever 如 `bge-base-en-v1.5`），檢查 H4：predictor 能不能跨策略遷移。
- 換一個 base model，檢查結論是否穩定。
