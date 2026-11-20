# MuSiQue Recoverability Pilot：完整實驗細節

研究問題：第一輪 retrieval 結束後，只用當下看得到的資訊，能否預測在固定的 corpus、retriever、query 策略和預算下，再搜一輪能不能補足證據？

> 範圍：這是 **pilot**，用來驗證標籤能不能建、訊號存不存在，**不是**最終實驗。
> 尚未包含的部分見最後一節。

---

## 0. 一鍵重跑

```bash
# 1) 環境（Python 3.12）
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch wheel 需配合該機器的 CUDA；見第 1 節

# 2) 先跑 50 題 smoke test（約 2 分鐘，含模型下載）
bash run_all.sh out_smoke 50
#    預期 run.log 最後一行：
#    Counter({'UNREC': 23, 'RECOV': 21, 'SUFF1': 6}) Counter({'UNREC': 41, 'SUFF1': 6, 'RECOV': 3})

# 3) 全量 2417 題（RTX 3080 約 8 分鐘）
bash run_all.sh out_full

# 4) 和參考結果比對
diff <(cut -c1-200 out_full/labels.jsonl) <(cut -c1-200 reference_results/labels.jsonl) | head
cat out_full/feature_auc.json reference_results/feature_auc.json
```

環境變數：`DEVICE=cuda:0`（指定 GPU）、`BATCH=16`、`PY=python`。

---

## 1. 硬體與軟體

| 項目 | 參考跑的設定 |
|---|---|
| GPU | 1× RTX 3080 10GB（只用一張卡） |
| VRAM 需求 | batch 16 可以跑；**batch 32 在 10GB 會 OOM**。24GB 以上的卡可試 batch 32–64 |
| RAM | 約 3GB |
| Python | 3.12.3 |
| 套件 | 見 `requirements.txt`：torch 2.15.0、transformers 5.23.0、bm25s 0.3.16、PyStemmer 3.1.0、scikit-learn 1.9.1 |
| CUDA | driver 595.99，torch cu132 wheel。其他機器請安裝符合該機器 CUDA 的 torch，版本不同不影響 BM25 部分 |
| 網路 | 需要連 huggingface.co，下載資料集（30MB）和模型（約 6GB） |

---

## 2. 資料

| 項目 | 值 |
|---|---|
| 資料集 | MuSiQue-Ans v1.0 **dev**，2417 題（2-hop 1252、3-hop 760、4-hop 405） |
| 來源 | `https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_ans_v1.0_dev.jsonl` |
| SHA-256 | `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`（`run_all.sh` 會自動檢查） |
| 用到的欄位 | `question`、`answer`、`answer_aliases`、`paragraphs[].{title, paragraph_text, is_supporting}`、`question_decomposition`（只用在核對） |

### Corpus 建構（所有題目共用同一個固定 corpus）
- 把 dev 集**所有題目的所有段落**（每題 20 段）合併。
- 用 `md5(title + "\n" + paragraph_text)` 去重，得到 **21,100** 段。
- 每段的索引文字是 `title + " " + paragraph_text`。
- 每題的 gold = 該題 `is_supporting=True` 的段落在 corpus 中的 id。2-hop 有 2 段，3-hop 有 3 段，4-hop 有 4 段。

注意：MuSiQue 的干擾段落當初就是用 BM25 挑出來的，所以 BM25 在這裡特別吃虧。這是刻意保留的難度。

---

## 3. 檢索流程

### Retriever（兩輪相同）
- `bm25s.BM25()` 預設參數：`method="lucene"`、`k1=1.5`、`b=0.75`。
- 分詞：`bm25s.tokenize(..., stopwords="en", stemmer=PyStemmer.Stemmer("english"))`。

### Round 1（決策模組只能看到這一輪）
- query = 原始問題。
- 保留 top-30（只用來算特徵和 k=10 對照），**送進上下文的是 top-5**（`K1=5`）。

### Round 2（只用來建立標籤）
- **Query 生成**：
  - 模型：`Qwen/Qwen2.5-3B-Instruct`，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，bf16。
  - 解碼：greedy（`do_sample=False`），`max_new_tokens=32`，batch 16，左側 padding。
  - 後處理：取輸出的第一行，去掉首尾的 `"`。
- **Prompt**（用 chat template 包成單一 user turn）：
  ```
  You are helping answer a multi-hop question using a search engine over Wikipedia paragraphs.
  Question: {q}

  Passages retrieved so far:
  [1] {title}: {text}
  ...
  [5] {title}: {text}

  What piece of information is still missing to answer the question? Write ONE short keyword search query that would find it. If an intermediate entity is already known from the passages, use its name in the query. Output only the query, nothing else.
  ```
- **檢索**：BM25 用生成的 query 取 top-20，**排除 round 1 已經取過的 5 篇**，再取前 5 篇（`K2=5`）。
- **預算**：每題 2 次 BM25 呼叫、1 次 LLM 呼叫，最多看 10 篇文件。

---

## 4. 標籤規則

令 `S` 為該題的 gold 段落集合，`R1` 為 round-1 的 top-5，`R2` 為 round-2 新取得的 5 篇。

| 標籤 | 條件 |
|---|---|
| `SUFF1`（首輪已足夠） | `S ⊆ R1` |
| `RECOV`（第二輪能補足） | 不是 SUFF1，且 `S ⊆ R1 ∪ R2` |
| `UNREC`（第二輪仍不足） | 其餘 |

**對照標籤 `label_k10_control`**：不寫新 query，直接取原始問題的 top-10（文件預算相同）。這組用來檢查「能補足」是靠改寫 query，還是只是多取了幾篇。

**這是 proxy 標籤**：用 gold 段落覆蓋代替「證據是否足以作答」，沒有呼叫 generator 去作答。

---

## 5. 決策時可用的特徵（`features.py`）

只使用：問題、round-1 排名和 BM25 分數、round-1 段落文字，以及（`q2` 組）LLM 寫出的 query 文字本身。**不使用** round-2 的檢索結果。

| 群組 | 特徵 |
|---|---|
| relevance | `top1`：round-1 第一名的 BM25 分數。這就是 brief 裡的 relevance-threshold baseline |
| score stats | `top1`、`gap12`（第 1 名減第 2 名）、`mean5`、`r5_10`（第 5 名減第 10 名）、`nqc`（top-30 的 std/mean）、`qlen`（問題內容詞數） |
| lexical | `cov_union`（top-5 合併後覆蓋問題詞的比例）、`cov_max`、`cov_min`、`multi`（被 ≥2 篇覆蓋的問題詞比例）、`titles_in_q`（標題出現在問題中的篇數） |
| q2 text | `q2_novel`（新 query 中不在原問題裡的詞的比例）、`q2_from_r1`（這些新詞出現在 round-1 段落中的比例）、`q2_len` |

**分類器**：`StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000)`，5-fold stratified CV，3 個 seed（0、1、2）取平均 AUC。

**兩個任務**：
1. SUFF1 vs. 其他（n=2417）
2. RECOV vs. UNREC，只在非 SUFF1 的題目上（n=2200）

另外跑一組只看 2-hop 的檢查，排除 hop 數混淆。

---

## 6. 參考結果（`reference_results/`）

### 標籤分布

| | SUFF1 | RECOV | UNREC |
|---|---|---|---|
| 全部（2417） | 217 | 334 | 1866 |
| 2-hop（1252） | 206 | 296 | 750 |
| 3-hop（760） | 11 | 35 | 714 |
| 4-hop（405） | 0 | 3 | 402 |
| 同 query 取 k=10 的對照 | 217 | **125** | 2075 |

- 334 題 RECOV 中，只有 50 題靠「同 query 多取 5 篇」也能補足。**大多數補足是改寫 query 帶來的**。
- 3-hop 和 4-hop 幾乎全是 UNREC，因為 2 輪 × 5 篇在結構上很難覆蓋 3–4 段 gold。正式實驗應**只用 2-hop**，或讓預算隨 hop 數增加。

### 只用第一輪資訊的預測力（AUC，3 seeds，std ≤ 0.001）

| 特徵群組 | SUFF1 vs. 其他 | RECOV vs. UNREC |
|---|---|---|
| relevance（top1） | 0.578 | 0.537 |
| score stats | 0.728 | 0.750 |
| + lexical | 0.775 | 0.749 |
| + q2 text | **0.783** | **0.775** |
| 只看 2-hop（全部特徵） | — | 0.705 |

### 延遲（RTX 3080，batch 16，每題平攤）

| 步驟 | 秒 / 題 |
|---|---|
| BM25 round 1 | 0.0004 |
| LLM 生成 query | 0.18（batched；單題未 batch 的延遲還沒量） |
| BM25 round 2 | 0.0006 |

### 18 題人工核對（`audit_notes_musique.md`）
- 抽樣：`sample_audit.py`，seed 7，每類 6 題（5 題 2-hop、1 題 3-hop）。
- **核對者是 Claude，不是人類標註員**。
- 以 gold 覆蓋判斷，標籤 18/18 正確。
- 6 題 UNREC 中，5 題的原因是 3B 模型寫錯第二輪 query，1 題是題目本身沒有指涉對象。**UNREC 很大程度反映 query 生成器的品質**。

---

## 7. 輸出檔案格式

`labels.jsonl`：每行一題。

| 欄位 | 說明 |
|---|---|
| `id`、`hops`、`question`、`answer`、`aliases` | 原始資料 |
| `supp` | gold 段落的 corpus id |
| `r1`、`r1_scores` | round-1 top-5 的 id 和 BM25 分數 |
| `r1_top30`、`r1_top30_scores` | round-1 top-30 |
| `q2` | LLM 生成的第二輪 query |
| `r2` | round-2 新取得的 5 篇 id |
| `cov1`、`cov12`、`cov_r1k10`、`n_supp` | 各情況下覆蓋到的 gold 數 |
| `label`、`label_k10_control` | 標籤 |
| `decomp` | MuSiQue 原始拆解，**只用在核對** |

其他輸出：
- `corpus.json`：corpus id → `{title, text}`。
- `timing.json`：延遲紀錄。
- `feature_auc.json`、`features.log`：AUC 結果。
- `audit_sample.txt`：18 題核對用的樣本。
- `run.log`：執行紀錄。

---

## 8. 重現性注意事項

- 本機驗證過：50 題 smoke test 的 label 和 q2 與全量跑的前 50 題**完全一致**（同 GPU、同 batch 16）。
- **換 GPU 型號、batch size 或 torch 版本**時，bf16 加左側 padding 的 greedy 輸出可能有少數 token 不同，導致個別題目的 q2 和標籤改變。分布和 AUC 應該接近，但不保證逐題相同。要逐題比對的話，請固定 batch 16 和相同的套件版本。
- BM25 和特徵部分是決定性的。

---

## 9. 尚未完成（不要當成已驗證）

1. **沒有 held-out test**：目前只有 CV。正式實驗需要切 validation 和 test，在 validation 調 threshold，在 test 報告。
2. **Brief 要求的 baseline 還沒做**：Sufficient Context 官方 prompt、固定再搜一輪、RINSE / S²G 類型的方法。
3. **沒有呼叫 generator 作答**：還沒有「無證據作答率」和「過早停止率」的取捨曲線。
4. **只有一種第二輪策略，而且偏弱**：需要加一種更強的策略（例如 7B 模型，或用 oracle 拆解寫 query），才能區分「策略失敗」和「corpus 裡真的沒有」。
5. **LongMemEval（長期記憶 / companion）版本還沒實作**。
