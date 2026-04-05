import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from app.services.rag_service import build_vector_store


if __name__ == "__main__":
    store = build_vector_store()
    print("Vector store built successfully.")
    print(f"Indexed documents: {store.index.ntotal}")
