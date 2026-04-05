from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from app.utils.config import get_settings


@dataclass
class RetrievalBundle:
    context: str
    docs: List[Document]
    strategy: str


class LocalBM25Index:
    def __init__(self, docs: Sequence[Document]) -> None:
        self.docs = list(docs)
        self.tokenized_docs = [doc.page_content.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, k: int = 2) -> List[Document]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [self.docs[idx] for idx, _score in ranked[:k]]


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set your Gemini API key.")

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=768,
    )


def _data_files() -> List[Path]:
    data_dir = get_settings().data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(data_dir.glob("*.txt"))


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


def split_documents(docs: Iterable[Document]) -> List[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(list(docs))


def build_vector_store() -> FAISS:
    settings = get_settings()
    chunks = split_documents(load_raw_documents())
    vector_store = FAISS.from_documents(chunks, get_embeddings())
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(settings.vector_store_path))

    manifest = {
        "chunk_count": len(chunks),
        "sources": sorted({chunk.metadata.get("source", "unknown") for chunk in chunks}),
    }
    (settings.vector_store_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return vector_store


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


def vector_store_ready() -> bool:
    settings = get_settings()
    return (settings.vector_store_path / "index.faiss").exists()


def _load_bm25_index() -> LocalBM25Index:
    chunks = split_documents(load_raw_documents())
    return LocalBM25Index(chunks)


def dedupe_documents(docs: Sequence[Document]) -> List[Document]:
    seen = set()
    unique_docs: List[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source", "unknown"), doc.page_content)
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def hybrid_retrieve(query: str, semantic_k: int | None = None, lexical_k: int = 2) -> RetrievalBundle:
    settings = get_settings()
    semantic_k = semantic_k or settings.retriever_k

    vector_store = load_vector_store()
    semantic_docs = vector_store.max_marginal_relevance_search(
        query,
        k=semantic_k,
        fetch_k=settings.mmr_fetch_k,
    )

    bm25_docs = _load_bm25_index().search(query, k=lexical_k)
    docs = dedupe_documents([*semantic_docs, *bm25_docs])

    context_parts = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Doc {idx} | {source}]\n{doc.page_content}")

    return RetrievalBundle(
        context="\n\n".join(context_parts),
        docs=docs,
        strategy="hybrid_mmr_bm25",
    )


def retrieve_docs(query: str) -> str:
    bundle = hybrid_retrieve(query)
    return bundle.context
