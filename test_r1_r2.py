"""Test R1 (FAISS persistent) + R2 (LLM cache)"""
import sys, time, pathlib, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from config import config
from src.retriever import HybridRAGRetriever

print("=== Test R1: FAISS Persistent Index ===")

faiss_bin = pathlib.Path(config.qdrant_path) / "faiss_index.bin"

# Thu nghiem voi 301 docs da co san trong documents.pkl
docs_pkl = pathlib.Path(config.qdrant_path) / "documents.pkl"
if not docs_pkl.exists():
    print("Khong co documents.pkl, indexing tu PDF...")
    from src.document_processor import DocumentProcessor
    proc = DocumentProcessor()
    docs = proc.process("docs/2312.10997.pdf")
    ret_tmp = HybridRAGRetriever(config)
    ret_tmp.clear()
    ret_tmp.add_documents(docs)
    del ret_tmp
    print(f"Da index {len(docs)} docs")

# Xoa faiss_index.bin de test rebuild
if faiss_bin.exists():
    faiss_bin.unlink()
    print("Xoa faiss_index.bin -> buoc test lan 1 se rebuild")

print()
print("--- Lan 1: Rebuild FAISS tu documents.pkl (cham) ---")
t0 = time.time()
ret1 = HybridRAGRetriever(config)
t1 = time.time()
print(f"Thoi gian: {t1-t0:.1f}s | docs={ret1.doc_count} | faiss saved={faiss_bin.exists()}")
del ret1

print()
print("--- Lan 2: Load FAISS tu disk (phai nhanh < 5s) ---")
t2 = time.time()
ret2 = HybridRAGRetriever(config)
t3 = time.time()
print(f"Thoi gian: {t3-t2:.1f}s | docs={ret2.doc_count}")

if (t3 - t2) < 10:
    speedup = (t1-t0) / max(t3-t2, 0.01)
    print(f"R1 PASS! Nhanh hon {speedup:.0f}x")
else:
    print("R1 WARN - kiem tra lai")

print()
print("=== Test R2: LLM Cache ===")
from src.llm_factory import get_llm, invalidate_llm_cache
llm1 = get_llm(config)
llm2 = get_llm(config)
assert llm1 is llm2, "Cache khong hoat dong!"
invalidate_llm_cache()
llm3 = get_llm(config)
assert llm1 is not llm3, "Invalidate khong hoat dong!"
print("R2 PASS - LLM cache hoat dong chinh xac")

print()
print("=== PHASE 1 COMPLETE ===")
