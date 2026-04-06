# Retrieval pipeline for local research documents.
# This module handles document loading, chunking, vector indexing,
# BM25 lexical retrieval, cache-aware hybrid retrieval, and context assembly.
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from app.services.cache_service import normalize_query
from app.utils.config import get_settings


@dataclass
# Structured retrieval output passed to tools and benchmarks.
class RetrievalBundle:
    context: str
    docs: List[Document]
    strategy: str
    cache_hit: bool = False


@dataclass(frozen=True)
# Snapshot of the retrieval cache counters for diagnostics or benchmarking.
class RetrievalCacheStats:
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


# Lightweight in-memory BM25 index built from the local text chunks.
class LocalBM25Index:
    def __init__(self, docs: Sequence[Document]) -> None:
        self.docs = list(docs)
        self.tokenized_docs = [doc.page_content.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, k: int = 2) -> List[Document]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [self.docs[idx] for idx, _score in ranked[:k]]


# Reuse a single embedding client instance across the process.
@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set your Gemini API key.")

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=768,
    )


# Discover source documents stored under the local data directory.
def _data_files() -> List[Path]:
    data_dir = get_settings().data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(data_dir.glob("*.txt"))


# Load source files once and attach the filename as metadata for citations/debugging.
@lru_cache(maxsize=1)
def load_raw_documents() -> List[Document]:
    docs: List[Document] = []
    for file_path in _data_files():
        loader = TextLoader(str(file_path), encoding="utf-8")
        for doc in loader.load():
            doc.metadata["source"] = file_path.name
            docs.append(doc)
    if not docs:
        raise FileNotFoundError("No .txt files found in data/. Add at least one research document.")
    return docs


# Cache chunking output because it is deterministic for a fixed corpus and config.
@lru_cache(maxsize=1)
def split_documents_cached() -> List[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(load_raw_documents())


# Non-cached splitter kept for flexibility in tests or future extensions.
def split_documents(docs: Iterable[Document]) -> List[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(list(docs))


# Build a FAISS index and write a small manifest for inspection.
def build_vector_store() -> FAISS:
    settings = get_settings()
    chunks = split_documents_cached()
    vector_store = FAISS.from_documents(chunks, get_embeddings())
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(settings.vector_store_path))

    manifest = {
        "chunk_count": len(chunks),
        "sources": sorted({chunk.metadata.get("source", "unknown") for chunk in chunks}),
    }
    (settings.vector_store_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    clear_retrieval_caches(clear_indexes=False)
    return vector_store


# Load the persisted FAISS index, or create it on first run.
@lru_cache(maxsize=1)
def load_vector_store() -> FAISS:
    settings = get_settings()
    index_file = settings.vector_store_path / "index.faiss"
    if not index_file.exists():
        return build_vector_store()

    return FAISS.load_local(
        str(settings.vector_store_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# Health-check helper used by the API layer.
def vector_store_ready() -> bool:
    settings = get_settings()
    return (settings.vector_store_path / "index.faiss").exists()


# Build the lexical retriever once, then reuse it for subsequent queries.
@lru_cache(maxsize=1)
def _load_bm25_index() -> LocalBM25Index:
    return LocalBM25Index(split_documents_cached())


# Remove duplicate chunks after combining semantic and lexical retrieval results.
def dedupe_documents(docs: Sequence[Document]) -> List[Document]:
    seen = set()
    unique_docs: List[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source", "unknown"), doc.page_content)
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


# Convert retrieved chunks into a readable context block for the agent/tool layer.
def _compose_context(docs: Sequence[Document]) -> str:
    context_parts = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Doc {idx} | {source}]\n{doc.page_content}")
    return "\n\n".join(context_parts)


# Internal cache over the hybrid retrieval result for repeated normalized queries.
@lru_cache(maxsize=256)
def _cached_hybrid_retrieve(normalized_query: str, semantic_k: int, lexical_k: int) -> RetrievalBundle:
    vector_store = load_vector_store()
    settings = get_settings()

    semantic_docs = vector_store.max_marginal_relevance_search(
        normalized_query,
        k=semantic_k,
        fetch_k=settings.mmr_fetch_k,
    )

    bm25_docs = _load_bm25_index().search(normalized_query, k=lexical_k)
    docs = dedupe_documents([*semantic_docs, *bm25_docs])

    return RetrievalBundle(
        context=_compose_context(docs),
        docs=docs,
        strategy="hybrid_mmr_bm25",
        cache_hit=False,
    )


# Expose retrieval cache metrics for diagnostics and benchmarking.
def get_retrieval_cache_stats() -> RetrievalCacheStats:
    info = _cached_hybrid_retrieve.cache_info()
    return RetrievalCacheStats(
        hits=info.hits,
        misses=info.misses,
        maxsize=info.maxsize,
        currsize=info.currsize,
    )


# Clear retrieval caches; optionally clear heavyweight index/chunk caches as well.
def clear_retrieval_caches(clear_indexes: bool = True) -> None:
    _cached_hybrid_retrieve.cache_clear()
    if clear_indexes:
        load_vector_store.cache_clear()
        _load_bm25_index.cache_clear()
        split_documents_cached.cache_clear()
        load_raw_documents.cache_clear()
        get_embeddings.cache_clear()


# Public retrieval entry point used by tools and benchmark scripts.
def hybrid_retrieve(query: str, semantic_k: int | None = None, lexical_k: int = 2) -> RetrievalBundle:
    settings = get_settings()
    semantic_k = semantic_k or settings.retriever_k
    normalized_query = normalize_query(query)

    before = _cached_hybrid_retrieve.cache_info()
    bundle = _cached_hybrid_retrieve(normalized_query, semantic_k, lexical_k)
    after = _cached_hybrid_retrieve.cache_info()
    cache_hit = after.hits > before.hits

    return RetrievalBundle(
        context=bundle.context,
        docs=list(bundle.docs),
        strategy=bundle.strategy,
        cache_hit=cache_hit,
    )


# Convenience helper returning only the assembled text context.
def retrieve_docs(query: str) -> str:
    bundle = hybrid_retrieve(query)
    return bundle.context
