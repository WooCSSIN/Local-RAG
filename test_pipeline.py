"""Quick test script — chạy: python test_pipeline.py"""
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)

from src.retriever import HybridRAGRetriever
from config import config

ret = HybridRAGRetriever(config)
print(f"Loaded from cache: {ret.doc_count} docs")

results = ret.retrieve("What is RAG?", k=3)
print(f"Retrieved: {len(results)} docs (with reranker)")
for i, r in enumerate(results):
    page = r.metadata.get("page", "?")
    print(f"  [{i+1}] page={page} | {r.page_content[:90]}...")

print("\nAll OK!")
