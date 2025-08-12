# rag_agent/knowledge_base.py
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KB_FILE = os.path.join(ROOT, 'data', 'handling_rules.txt')

def ensure_kb():
    if not os.path.exists(KB_FILE):
        with open(KB_FILE, 'w') as f:
            f.write("Fragile items: slow speed, avoid fast turns.\n")
            f.write("Heavy items (>20kg): require two robots and check battery > 50%.\n")
            f.write("Electronics: keep dry and avoid magnetic interference.\n")

def query_kb(query):
    """
    Very simple retrieval: returns lines that match any keyword from query.
    """
    ensure_kb()
    q = query.lower()
    results = []
    with open(KB_FILE, 'r') as f:
        for line in f:
            if any(word in line.lower() for word in q.split()):
                results.append(line.strip())
    return results if results else ["No exact rule found; follow standard handling procedures."]
