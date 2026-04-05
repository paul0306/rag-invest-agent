from app.services.rag_service import dedupe_documents
from langchain_core.documents import Document


def test_dedupe_documents() -> None:
    docs = [
        Document(page_content="abc", metadata={"source": "a.txt"}),
        Document(page_content="abc", metadata={"source": "a.txt"}),
        Document(page_content="xyz", metadata={"source": "b.txt"}),
    ]
    result = dedupe_documents(docs)
    assert len(result) == 2
