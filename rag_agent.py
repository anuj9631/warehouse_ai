# Minimal RAG scaffold using chromadb and local text files
from chromadb.utils import embedding_functions
import chromadb
import os

KB_DIR = os.path.join(os.path.dirname(__file__), '..', 'kb')
os.makedirs(KB_DIR, exist_ok=True)
KB_FILE = os.path.join(KB_DIR, 'handling_rules.txt')

def ensure_kb():
    if not os.path.exists(KB_FILE):
        with open(KB_FILE, 'w') as f:
            f.write("Fragile items: use slow speed, two-robot lift if weight > 5kg.\n")
            f.write("Heavy items (>20kg): use two robots and check battery > 50%.\n")
            f.write("Electronics: avoid magnetic fields and keep dry.\n")

def build_and_query(query):
    ensure_kb()
    # Using chromadb in-memory client
    client = chromadb.Client()
    collection = client.create_collection("kb")
    # naive embeddings with built-in text-embedding-3-small? (if you have no embedding model, use dummy)
    # We'll just store text as metadata and do naive string search for now (since embeddings require model)
    # This is a scaffold — replace with a real embedder if available.
    with open(KB_FILE, 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
    for i, d in enumerate(docs):
        collection.add(documents=[d], metadatas=[{"source": KB_FILE}], ids=[str(i)])
    # naive retrieval: return docs that contain word from query
    results = []
    q_words = set(query.lower().split())
    for d in docs:
        if any(w in d.lower() for w in q_words):
            results.append(d)
    return results or docs[:2]

if __name__ == "__main__":
    print(build_and_query("fragile heavy"))
