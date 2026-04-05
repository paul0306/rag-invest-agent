# Agentic RAG Investment Research Assistant

FastAPI + LangChain + Gemini API project for equity-research style Q&A.

## Features

- Agentic tool routing with LangChain `create_agent`
- Gemini chat model and Gemini embeddings
- FAISS vector store for local semantic retrieval
- Hybrid retrieval with MMR + BM25
- Simple local news tool so the project runs without extra API keys
- Benchmark script for retrieval latency

## Project structure

```text
rag-invest-agent/
├── app/
├── data/
├── scripts/
├── tests/
├── vector_store/
├── requirements.txt
└── .env.example
```

## Quick start

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Configure Gemini API key

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Build the FAISS index

```bash
python scripts/build_index.py
```

### 4. Run the API

```bash
python -m uvicorn app.main:app --reload
```

### 5. Test the endpoint

```bash
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d '{"query": "Analyze NVIDIA investment risk over the next 6 months."}'
```

## Example response shape

- Bullish factors
- Bearish risks
- Final summary
- Caveats

## Notes

- `news_search` is intentionally mocked so you can run the project with only a Gemini key.
- Replace `app/services/news_service.py` with a live adapter later if you want real-time headlines.
- The retrieval layer uses MMR for diversity and BM25 for lexical matching.

## Resume bullets

- Built an **agentic RAG investment research assistant** using **FastAPI, LangChain, Gemini API, and FAISS**
- Implemented **hybrid retrieval (MMR semantic search + BM25 lexical search)** to improve relevance and reduce redundant context
- Added a modular tool-routing architecture for **document retrieval, news summarization, and risk analysis**
- Benchmarked retrieval latency with a standalone script to support README metrics and performance tuning

## Tests

```bash
PYTHONPATH=. pytest
```
