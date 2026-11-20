# 交接文件：Evidence Recoverability in RAG

最後更新：2026-11-20

## 一句話狀態

Stage 1 已完成：建標籤，並檢查只用第一輪資訊能否預測第二輪的收益。
- **Wikipedia（MuSiQue）**：猜得到（AUC 0.775）。
- **長期記憶（LongMemEval、LoCoMo）**：猜不到（AUC 約 0.5）。

**主要嫌疑**：第二輪的搜尋詞是由小模型 Qwen2.5-3B 寫的，它常常寫錯。所以很多「第二輪也找不到」是**搜尋詞寫錯**，不是記憶裡真的找不到。這讓標籤失真，下一輪要用大模型處理。

---

## 1. Repo 狀態（⚠️ 先處理）

- 所有檔案都已 `git add`，**但還沒有 commit**，因為這台機器沒有設定 git 身分。
- 要 commit 和 push 時：
  ```bash
  git config user.name "<name>" && git config user.email "<email>"
  git commit -m "Recoverability pilot: MuSiQue + long-term memory"
  git branch -M main && git remote add origin <repo-url> && git push -u origin main
  ```
- 新 server 上：
  ```bash
  git clone <repo-url> && cd <repo>
  python3.12 -m venv .venv && source .venv/bin/activate
  pip install -r memory_poc/requirements.txt   # torch 請裝符合該機器 CUDA 的版本
  bash memory_poc/run_all.sh out_mem_smoke 40  # smoke test
  ```

| 資料夾 | 內容 |
|---|---|
| `memory_poc/` | **主線**：LongMemEval_S + LoCoMo。完整設定見 `memory_poc/README.md` |
| `poc/` | 對照：MuSiQue。完整設定見 `poc/README.md` |
| `*/reference_results/` | 本機（RTX 3080，Qwen2.5-3B）的參考結果，用來比對 |

---

## 2. 研究問題與標籤（白話）

**問題**：系統搜完第一輪、發現證據不夠時，能不能在搜第二輪**之前**，先猜到「第二輪會不會找到缺的證據」？

| 標籤 | 意思 | 理想行動 |
|---|---|---|
| SUFF1 | 第一輪就找到全部 gold 證據 | 直接回答 |
| RECOV | 第一輪沒找齊，加上第二輪就找齊了 | 再搜 |
| UNREC | 兩輪都沒找齊 | 換方法，或暫緩作答 |
| NOANS | 記憶裡沒有答案：原生 abstain 題，或反事實刪除（cfdel）造出來的題 | 問使用者 |
| ADVERS | LoCoMo cat5：記憶裡有很像的內容，但說的是別人 | 不能被騙 |

**規則**：預測器只能看第一輪的結果。第二輪只用來產生標籤。

---

## 3. 目前結果

### 預測「第二輪有沒有用」（RECOV vs. UNREC），AUC：0.5 = 亂猜

| | MuSiQue | LongMemEval_S | LoCoMo |
|---|---|---|---|
| 只用 top-1 分數（relevance baseline） | 0.537 | 0.549 | 0.552 |
| 全部淺層特徵 | **0.775** | **0.568** | **0.496** |

### 標籤分布

| | SUFF1 | RECOV | UNREC | NOANS | ADVERS |
|---|---|---|---|---|---|
| MuSiQue（2417） | 217 | 334 | 1866 | — | — |
| LongMemEval_S（970，含 470 cfdel） | 330 | 42 | 98 | 500 | — |
| LoCoMo（3521，含 1535 cfdel） | 797 | 117 | 621 | 1535 | 446 |

---

## 4. ⭐ 核心發現：小模型寫的搜尋詞品質差，讓標籤失真

### 證據
1. **MuSiQue 人工核對**（核對者是 Claude，不是人類）：6 題 UNREC 中，**5 題是搜尋詞寫錯**，例如抄了干擾段落的人名、重複第一跳、憑空編出實體。只有 1 題是題目本身有問題。
2. **搜尋詞 ≈ 照抄原問題的比例**（token Jaccard ≥ 0.6）：

   | | SUFF1 | RECOV | UNREC |
   |---|---|---|---|
   | MuSiQue | 5% | 4% | 9% |
   | LongMemEval | 27% | 26% | 34% |
   | LoCoMo | **51%** | **55%** | **45%** |

   在記憶情境，約 1/3 到一半的「第二輪」其實是**用原問題再搜一次**，等於只是多取幾篇。
3. **對照組也吻合**：RECOV 中，原問題取深一點也能補足的比例，LongMemEval 是 25/42，LoCoMo 是 84/117。
4. **典型錯誤**：

   | 問題 | 3B 寫的搜尋詞 | 錯在哪 |
   |---|---|---|
   | 我生日去夏威夷打算住哪？ | 原問題原封不動 | 沒利用第一輪的線索 |
   | Frederic Ewen 的雇主屬於哪個機構？ | `Suetonius Grant Heatly employer` | 抄了不相關的人名 |
   | 冥界之王是什麼的一部分？ | `Romance of the Underworld characters…` | 被電影名稱帶偏 |

### 影響
- **UNREC 混了兩種東西**：(a) 搜尋詞寫錯，換個好的搜尋詞就找得到；(b) 真的難找，或者記憶裡沒有。
- 記憶情境的 AUC ≈ 0.5，可能只是因為標籤裡 (a) 太多，預測器在猜「小模型會不會寫錯」，而這從第一輪的結果看不出來。
- **在用更強的搜尋詞產生器重做標籤之前，不能下「記憶情境猜不到」的結論。**

---

## 5. 下一輪計畫（新 server 可以跑大模型）

### Step 1：換大模型寫第二輪搜尋詞（最優先，約 1–2 小時）
程式已經支援 `--device auto`，會把大模型分散到所有 GPU；本機用兩張卡驗證過。

```bash
# 例：Qwen2.5-72B-Instruct（bf16 約需 150GB VRAM）或 32B（約 70GB）
MODEL=Qwen/Qwen2.5-72B-Instruct DEVICE=auto BATCH=8 bash memory_poc/run_all.sh out_mem_72b
MODEL=Qwen/Qwen2.5-72B-Instruct DEVICE=auto BATCH=8 bash poc/run_all.sh out_musique_72b
```

要回報的數字：
- 標籤變化矩陣（3B 的標籤 → 72B 的標籤），特別是 **UNREC → RECOV 的數量**。這直接量化「搜尋詞寫錯」佔了多少。
- 搜尋詞照抄原問題的比例。
- 重跑 `features_memory.py`，看 T2 的 AUC 有沒有上升。

### Step 2：把 UNREC 拆成「策略失敗」和「真的找不到」（規格，尚未實作）
- **UNREC-strategy**：3B 找不到，但大模型的搜尋詞找到了。
- **UNREC-hard**：大模型也找不到，K2 加大到 50 也找不到。
- MuSiQue 還可以加一個 oracle 上界：用 `question_decomposition` 的第二跳問題，把第一跳的答案代進去當搜尋詞。

### Step 3：標籤改成「實際作答是否答對」（規格，尚未實作）
- Generator：大模型。分別在 R1 和 R1∪R2 的條件下作答，允許拒答。
- Judge：LLM judge。LongMemEval 官方 repo 有評分 prompt，可以拿來用，把模型換成開源大模型。
- 新的標籤定義：
  - SUFF1 = R1 下答對
  - RECOV = R1 下答錯、R1∪R2 下答對
  - UNREC = 兩者都答錯
  - NOANS 題答對的定義是「有拒答或追問」
- 這可以處理 gold 標註的雜訊：例如 Oahu 那題答案其實搜到了，Jon 和 Gina 那題要湊 5 個 turn 太嚴格。

### Step 4：較強的預測器與 baseline（規格，尚未實作）
- LLM 讀第一輪結果，判斷「再搜有沒有用」。
- Sufficient Context 官方 autorater prompt（`github.com/hljoren/sufficientcontext`）。
- Baseline：永遠再搜、永遠不搜、relevance threshold。

### Step 5：端到端成本（規格，尚未實作）
- 在相同的無證據作答率下，比較延遲（包含預測器本身）、檢索次數，以及過早停止率。過早停止 = RECOV 卻被判成不搜。

---

## 6. 已知問題與陷阱

| 問題 | 狀態 |
|---|---|
| **重現性**：換 batch 組成或 GPU 數量，q2 會有少數不同 | 實測 `--device auto`（兩張卡、沒有 cfdel）和參考結果比：q2 相同 7/10，標籤相同 10/10。要逐題比對請固定 batch 和 GPU 配置 |
| 記憶大小特徵會洩漏 cfdel | ✅ 已從特徵中移除。單獨使用時 AUC 0.679 |
| cfdel 刪不乾淨 | ⚠️ LongMemEval：354 題中有 15 題刪除後仍找得到答案字串。LoCoMo：別人重複同一句話，例如 Sam「progress over perfection」那題。需要改成連重複的 turn 一起刪 |
| gold 標註雜訊 | ⚠️ LoCoMo 有說話者標錯的例子（Harry Potter 是 Tim 說的，題目問 John）。Step 3 可以減輕 |
| 全覆蓋太嚴格 | ⚠️ 多跳題和 multi-session 題需要很多個 gold turn。Step 3 可以減輕 |
| LongMemEval 原生 abstain 只有 30 題 | NOANS 主要來自 cfdel，論文中必須寫明 |
| T3（RECOV vs. NOANS）在 LongMemEval 的 0.858 | 可能有一部分是構造方式造成的：整個 session 被刪掉。LoCoMo 只有 0.554 |

---

## 7. 程式 TODO（小）
- ✅ 兩個 `run_all.sh` 都支援 `MODEL=` 和 `DEVICE=auto`（已修好，並用 50 題重跑驗證）。
- 大模型可以考慮改用 vLLM 加速。目前用 transformers 的 `generate`，72B 會慢。

---

## 8. 文獻定位（已核實，細節見對話紀錄與各 README）
- **最接近的工作**：Stop-RAG（arXiv 2510.14337）用未來 rollout 建 value label；Tian et al.（CIKM'26，2609.16453）預測 partial utility；Luo（2608.13237）做 Search-R1 的停止判斷。
- **它們都沒做的**（我們的推論）：
  - 不可回答題
  - 把「再搜也補不回」當成一類
  - 把決策模組的成本算進端到端延遲
  - 記憶 / companion 情境
- **Brief 的更正**：Sufficient Context 的 repo 只有 prompt，沒有程式碼；RINSE、S²G-RAG、GRAB-RAG 都沒有公開程式碼。

---

## 9. 等使用者決定的事
1. 新 server 的 GPU 規格，決定用 32B 還是 72B。
2. Step 1 跑完之後，記憶情境的 T2 如果還是約 0.5，研究主問題是否改成「該再搜還是該追問」（T3）。
