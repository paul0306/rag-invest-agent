# Utility script for building the FAISS vector index from local documents.
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from app.services.rag_service import build_vector_store


# Build the vector index from the local corpus and print a small summary.
if __name__ == "__main__":
    store = build_vector_store()
    print("Vector store built successfully.")
    print(f"Indexed documents: {store.index.ntotal}")
