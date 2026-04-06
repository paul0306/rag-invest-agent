# Agentic RAG 投資研究助理

這是一個以 **FastAPI + LangChain + Gemini API + FAISS** 為核心的作品集專案，主題聚焦在 **投資研究 / 財報問答 / 風險分析**。專案使用 **Agentic RAG** 架構：先做文件檢索，再讓具備 tool routing 能力的 agent 根據問題內容決定要不要使用 **RAG 文件檢索、新聞摘要、風險分析** 等工具，最後生成結構化研究摘要。

這個專案的定位不是做成複雜的交易系統，而是做成一個 **適合放在履歷、GitHub 與面試中展示的 LLM backend 專案**：

- 有清楚的 **API 層**（FastAPI）
- 有清楚的 **Agent 層**（LangChain + Gemini）
- 有清楚的 **Retrieval 層**（FAISS + BM25 + MMR）
- 有清楚的 **效能優化與 benchmark**（cache / retrieval profiling）

---

## 一、專案亮點

### 1. Agentic RAG，而不只是單純問答
本專案不是把文件餵進 LLM 就結束，而是透過 LangChain agent 讓模型可以根據問題內容選擇不同工具：

- `rag_search`：查本地財報 / 研究文件
- `news_search`：查近期新聞摘要（目前為 mock 版本）
- `risk_analyzer`：補充啟發式風險清單

這讓系統能夠更接近實務上的「研究助理」流程，而不是單一步驟的 QA demo。

### 2. Hybrid Retrieval 設計
Retrieval 層結合了：

- **FAISS semantic retrieval**
- **MMR (Max Marginal Relevance)**：降低重複 chunk
- **BM25 lexical retrieval**：補 keyword match 能力

這種設計比純向量搜尋更適合金融 / 財報情境，因為財報與研究問題常常包含明確關鍵字、產品名、公司名與指標名詞。

### 3. 有做效能優化，不只是功能完成
專案額外補了 retrieval layer 的加速與 profiling：

- 向量索引預先建立
- chunk 結果快取
- BM25 index 快取
- hybrid retrieval cache

---

## 二、系統架構

```text
User Query
   ↓
FastAPI Endpoint
   ↓
LangChain Agent
   ↓
Tool Routing
 ┌─────────────────────────────┐
 │ rag_search                  │
 │ news_search                 │
 │ risk_analyzer               │
 └─────────────────────────────┘
   ↓
Gemini Final Answer Generation
```

### RAG 流程

```text
Query
  ↓
Normalize Query
  ↓
Hybrid Retrieval
  ├─ FAISS Semantic Search
  ├─ MMR Diversity Selection
  └─ BM25 Lexical Search
  ↓
Deduplicate Chunks
  ↓
Compose Context
  ↓
LLM / Agent
```

---

## 三、專案結構

```text
rag-invest-agent/
├── app/
│   ├── api/                 # FastAPI routes
│   ├── models/              # Pydantic schemas
│   ├── services/            # Agent / RAG / tools / cache / mock news
│   └── utils/               # Config and prompt templates
├── data/                    # Local research documents used for indexing
├── scripts/                 # Index build and benchmark scripts
├── tests/                   # Basic smoke tests
├── vector_store/            # Persisted FAISS index
├── requirements.txt
├── .env.example
└── README.md
```

---

## 四、技術棧

- **Backend API**：FastAPI
- **Agent Framework**：LangChain
- **LLM / Embedding**：Gemini API (`langchain-google-genai`)
- **Vector Store**：FAISS
- **Lexical Retrieval**：BM25 (`rank-bm25`)
- **Chunking**：RecursiveCharacterTextSplitter

---

## 五、主要模組說明

### `app/services/agent_service.py`
負責建立 Gemini chat model 與 LangChain agent，並提供 `run_analysis()` 當作整個系統的主要入口。

### `app/services/rag_service.py`
負責：

- 載入本地研究文件
- 做 chunking
- 建立與讀取 FAISS index
- 建立與快取 BM25 index
- 執行 hybrid retrieval
- 組出可以直接送進 LLM 的 context

### `app/services/tools.py`
定義 agent 可呼叫的工具：

- `rag_search`
- `news_search`

### `scripts/build_index.py`
一鍵建立向量索引。

### `scripts/benchmark.py`
量測：

- retrieval latency
- rag tool latency
- agent end-to-end latency
- warm cache speedup

---

## 六、如何啟動專案

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 設定 Gemini API Key

複製設定檔：

```bash
cp .env.example .env
```

在 `.env` 內填入：

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. 建立向量索引

```bash
python scripts/build_index.py
```

### 4. 啟動 FastAPI

```bash
python -m uvicorn app.main:app --reload
```

### 5. 打開 Swagger UI

啟動後可在瀏覽器查看：

```text
http://127.0.0.1:8000/docs
```

---

## 七、API 說明

### `GET /health`
回傳服務健康狀態與向量索引是否已準備完成。

範例回傳：

```json
{
  "status": "ok",
  "vector_store_ready": true
}
```

### `POST /analyze`
輸入投資研究問題，取得結構化分析結果。

範例 request：

```json
{
  "query": "Analyze NVIDIA investment risk over the next 6 months."
}
```

範例回應結構：

- Bullish factors
- Bearish risks
- Final summary
- Caveats