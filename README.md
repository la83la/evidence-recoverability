# Evidence Recoverability in RAG

第一輪檢索之後，能否預測再搜一輪能不能補足證據？

| 資料夾 | 內容 | 狀態 |
|---|---|---|
| `memory_poc/` | **主線**：長期記憶情境（LongMemEval_S + LoCoMo） | stage 1 完成（標籤 + 可分性檢查） |
| `poc/` | 對照：Wikipedia 多跳 QA（MuSiQue） | stage 1 完成 |
| `RESEARCH_BELIFE.md` | 原始研究 brief | — |

快速開始：
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r memory_poc/requirements.txt
bash memory_poc/run_all.sh out_mem_smoke 40   # smoke test
bash memory_poc/run_all.sh out_mem            # 全量
bash poc/run_all.sh out_musique                # MuSiQue 對照
```
完整細節見各資料夾的 README.md。
