# Gemma 2B 跨層連續提示迴圈魔改計畫

**目標：** 在 M2 MacBook Air (24GB) 環境下，使用 MLX 框架對 Gemma 2B 進行架構微調，透過「由上而下回饋」機制，賦予靜態語言模型內部反思能力。

---

## 專案總覽與架構定義

本專案的核心目標是打破 Transformer 單向推論的限制，讓模型能在最終輸出前，先進行一次內部的「預演與反思」。

### 數學邏輯定義

1. **第一次傳遞（提取高階特徵）：** 將輸入序列餵給模型，並提取最後一層的最後一個 Token 的隱藏層狀態。
   $$H_{top} = \text{Gemma}(X_{\text{input}})_{[-1]}$$

2. **轉譯器處理（生成虛擬代幣）：** 將提取出的高階狀態輸入自定義的 Adapter，轉換為連續提示詞（Soft Prompts）。
   $$P_{soft} = \text{Adapter}(H_{top})$$

3. **第二次傳遞（回饋與重新預測）：** 將生成的連續提示詞與原始輸入的 Embedding 向量進行拼接，並讓模型重新執行一次完整的 Forward Pass 來產生最終預測。
   $$X_{second} = \text{Concat}(P_{soft}, \text{Embedding}(X_{\text{input}}))$$
   $$Y_{pred} = \text{Gemma}(X_{second})$$

---

## Phase 1: 環境建置與基底準備 (Environment & Baseline)

本階段目標為確保開發環境穩定，並取得原始模型單向傳遞的效能基準線。

* **建置 MLX 生態系：** 安裝 Apple 官方的 `mlx` 與 `mlx-lm` 庫，確保 M2 晶片的 GPU 加速正常啟用。
* **載入權重與凍結：** 下載 Gemma 2B 的完整精度或 BF16 權重，並使用程式碼將 Transformer 主體結構的 `requires_grad` 全部設為 `False`。
* **建立基準線 (Baseline)：** 選定一個小型的邏輯推理資料集（例如 GSM8K 或自建的 QA 文本），讓原始的 Gemma 2B 進行單向預測，記錄下平均的 Cross-Entropy Loss 與記憶體峰值佔用量。

---

## Phase 2: 核心架構魔改 (Architecture Modification)

本階段為專案中最關鍵的工程挑戰，我們需要覆寫模型預設的 Forward Pass 流程。

* **攔截高階隱藏狀態：** 修改模型的推論函數，使其在處理完輸入字串後，不直接輸出預測字元，而是回傳最後一個層級、最後一個 Token 的 Hidden State 向量。
* **建構 Adapter 橋接器：** 使用 MLX 建立一個包含兩層線性映射與 GELU 激活函數的微型多層感知機（MLP）。其輸入維度為 Gemma 2B 的隱藏層維度，輸出維度為 `4 * 隱藏層維度`（代表 4 個虛擬代幣）。
* **實作拼接邏輯：** 將 Adapter 輸出的向量 Reshape 後，與原始輸入字串經過 Embedding Layer 產生的向量在序列維度上進行拼接（Concatenation）。
* **閉環測試：** 將拼接後的完整矩陣再次送入 Gemma 主體，確認第二次 Forward Pass 不會報錯，且維度完全吻合。

---

## Phase 3: 訓練迴圈建構 (Training Loop & Fine-tuning)

架構接通後，接下來要透過機器學習讓 Adapter 學會如何將高階語意「壓縮」成有用的提示。

* **定義損失函數：** 針對第二次 Pass 輸出的 Logits，與資料集中的真實下一個字（Ground Truth）計算 Cross-Entropy Loss。第一次 Pass 不計算任何 Loss。
* **實作反向傳播：** 利用 MLX 的 `mlx.core.value_and_grad` 功能，將計算出的梯度只傳遞並更新給 Adapter 的參數。
* **記憶體監控與 Batch Size 測試：** 在 24GB 的 M2 Air 上，從 Batch Size 為 1 開始測試，觀察訓練過程中的記憶體水位，確保系統不會啟動 Swap 導致降速。
* **執行微調：** 讓模型在資料集上跑幾個 Epoch，觀察 Loss 曲線是否有穩定下降的趨勢，這代表模型正在學習利用迴圈帶來的「預期資訊」。

---

## Phase 4: 評估驗證與推論優化 (Evaluation & Optimization)

模型訓練完成後，必須驗證大腦迴圈機制是否帶來了邏輯能力的提升。

* **對比測試：** 使用未見過的測試資料集，分別跑一次原始的單向生成，以及加入 Adapter 的雙向生成，記錄兩者的準確率與生成品質。
* **分析特徵空間：** 將 Adapter 生成的 4 個虛擬代幣向量印出來，計算它們與真實單字 Embedding 的餘弦相似度（Cosine Similarity），嘗試解讀模型到底「反思」了什麼內容。
* **封裝推論腳本：** 將這套雙次傳遞的邏輯寫成一個乾淨、獨立的 Python Class，方便未來直接匯入文本進行推論，並可考慮開源分享此實驗成果。

---

## 專案時程總覽

| 階段 | 核心目標 | 預期產出物 |
| :--- | :--- | :--- |
| **Phase 1** | 基底建置與測試 | 可執行的推論腳本、基準 Loss 數據 |
| **Phase 2** | 架構魔改與拼接 | 包含迴圈機制的自定義 Model Class |
| **Phase 3** | Adapter 微調訓練 | 訓練迴圈程式碼、Adapter 權重檔 |
| **Phase 4** | 效能對比與分析 | 驗證報告與優化後的推論腳本 |

---

## 專案目錄結構

```
gemma2b-continuous-reflection-mlx/
├── README.md                 本文件
├── requirements.txt          Python 套件需求 (mlx, mlx-lm ...)
├── scripts/
│   └── setup_env.sh          建立虛擬環境與安裝依賴
├── src/
│   ├── __init__.py
│   ├── baseline.py           Phase 1：原始 Gemma 2B 單向基準測試
│   ├── adapter.py            Phase 2：Adapter 橋接器 (MLP, GELU)
│   ├── model_wrapper.py      Phase 2：覆寫 forward pass，雙次傳遞與拼接邏輯
│   ├── train.py              Phase 3：訓練迴圈，只更新 Adapter 參數
│   └── evaluate.py           Phase 4：對比測試、Cosine Similarity 分析
├── data/                     資料集（如 GSM8K 子集、自建 QA 文本）
├── notebooks/                探索性實驗、視覺化
└── results/                  Loss 曲線、記憶體紀錄、評估報告
```

## 快速開始

```bash
cd gemma2b-continuous-reflection-mlx
bash scripts/setup_env.sh
source .venv/bin/activate
python src/baseline.py
```
