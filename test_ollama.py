"""
test_ollama.py — Kiểm tra Ollama + toàn bộ pipeline RAG
Chạy SAU KHI đã: ollama serve && ollama pull <model>

Usage:
    python test_ollama.py
    python test_ollama.py --model llama3.2:3b
"""
import sys
import argparse
import logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from config import config
from src.utils import check_ollama_connection, list_ollama_models
from src.retriever import HybridRAGRetriever

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="Override LLM model")
args = parser.parse_args()

if args.model:
    config.llm_model = args.model

print("=" * 50)
print("RAG Pipeline Test")
print("=" * 50)

# 1. Kiểm tra Ollama
print("\n[1] Kiểm tra Ollama...")
ok, msg = check_ollama_connection(config.llm_base_url)
print(f"    {msg}")
if not ok:
    print("\n❌ Ollama chưa chạy. Hãy chạy: ollama serve")
    sys.exit(1)

# 2. Kiểm tra model
print(f"\n[2] Kiểm tra model: {config.llm_model}")
models = list_ollama_models(config.llm_base_url)
if config.llm_model not in models:
    print(f"    ⚠️  Model chưa có. Đang pull {config.llm_model}...")
    import subprocess
    subprocess.run(f"ollama pull {config.llm_model}", shell=True)

# 3. Kiểm tra retriever
print(f"\n[3] Kiểm tra vector index...")
ret = HybridRAGRetriever(config)
print(f"    Docs: {ret.doc_count}")
if ret.doc_count == 0:
    print("    ⚠️  Index rỗng. Upload PDF qua app.py trước.")

# 4. Test LLM đơn giản
print(f"\n[4] Test LLM ({config.llm_model})...")
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=config.llm_model, base_url=config.llm_base_url, temperature=0)
    response = llm.invoke("Trả lời ngắn gọn: 2+2 bằng mấy?")
    print(f"    LLM OK → '{response.content.strip()}'")
except Exception as e:
    print(f"    ❌ LLM lỗi: {e}")
    sys.exit(1)

# 5. Test RAG end-to-end nếu có docs
if ret.doc_count > 0:
    print(f"\n[5] Test RAG end-to-end...")
    from src.rag_graph import build_rag_graph, make_initial_state
    graph = build_rag_graph(ret, config)
    state = make_initial_state("What is RAG? Give a brief answer.")
    result = graph.invoke(state)
    answer = result["answer"][:200]
    print(f"    Query: 'What is RAG?'")
    print(f"    Answer: {answer}...")
    print(f"    Sources: {len(result.get('context', []))} docs retrieved")
else:
    print(f"\n[5] Bỏ qua RAG test (index rỗng)")

print("\n" + "=" * 50)
print("✅ Tất cả OK! Chạy app: python app.py")
print("=" * 50)
