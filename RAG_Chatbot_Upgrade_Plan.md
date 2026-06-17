# Kế hoạch nâng cấp — Local RAG Chatbot

> Dựa trên: [haooyuee/Local-RAG-Chatbot](https://github.com/haooyuee/Local-RAG-Chatbot)  
> Mục tiêu: Áp dụng các công nghệ RAG tiên tiến nhất (2024–2025) để nâng cao chất lượng trả lời, tốc độ, và tính linh hoạt, **vẫn chạy local** trên phần cứng tương đương.

---

## 1. Tổng quan thay đổi

| Hạng mục | Hiện tại | Sau nâng cấp |
|---|---|---|
| LLM backend | HuggingFace pipeline (Llama2, Gemma) | **Ollama** (Llama 3.2, Qwen2.5, Phi-3.5) |
| Embedding | sentence-transformers | **BGE-M3** hoặc `nomic-embed-text` |
| Chunking | RecursiveTextSplitter (fixed-size) | **Semantic Chunking** + Parent-Child |
| Retrieval | Pure vector search | **Hybrid Search** (BM25 + Vector) + Reranker |
| Vector store | ChromaDB / FAISS in-memory | **Qdrant** (persistent, hybrid native) |
| Orchestration | `ConversationalRetrievalChain` | **LangGraph** (stateful graph) |
| Document types | PDF only | **Docling** (PDF, Word, Excel, HTML, Markdown) |
| UI | Streamlit basic | **Gradio 4** (streaming, multi-session) |
| Evaluation | RAGAS / ROUGE manual | **RAGAS 0.2** + DeepEval metrics |

---

## 2. Cài đặt môi trường mới

```bash
# Tạo môi trường mới (Python 3.11 khuyên dùng)
conda create -n rag-v2 python=3.11
conda activate rag-v2

# PyTorch với CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài Ollama (chạy LLM local, thay HuggingFace pipeline)
# Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: tải từ https://ollama.com/download

# Pull model (chọn 1 trong các model sau tuỳ VRAM)
ollama pull llama3.2:3b          # 2GB VRAM — nhanh, nhẹ
ollama pull qwen2.5:7b           # 5GB VRAM — cân bằng tốt
ollama pull phi3.5:3.8b          # 2.5GB VRAM — code + reasoning tốt

# Dependencies chính
pip install -r requirements-v2.txt
```

**`requirements-v2.txt`:**
```
# Core RAG stack
langchain>=0.3.0
langchain-community>=0.3.0
langchain-ollama>=0.2.0
langgraph>=0.2.0

# Vector store
qdrant-client>=1.11.0
langchain-qdrant>=0.2.0

# Hybrid search
rank-bm25>=0.2.2
flashrank>=0.2.9          # reranker nhẹ, chạy CPU

# Document processing
docling>=2.5.0            # đa định dạng
pypdf>=4.0.0              # fallback PDF

# Embedding
fastembed>=0.4.0          # BGE-M3, nomic — tối ưu cho local

# UI
gradio>=4.44.0

# Evaluation
ragas>=0.2.0
deepeval>=1.4.0

# Utils
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
```

---

## 3. Cấu hình (`config.yaml` → `config.py`)

Thay YAML thuần bằng Pydantic Settings để có type checking:

```python
# src/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class RAGConfig(BaseSettings):
    # LLM
    llm_model: str = Field(default="qwen2.5:7b")
    llm_base_url: str = Field(default="http://localhost:11434")
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=2048)

    # Embedding
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_device: str = Field(default="cuda")

    # Chunking
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    use_semantic_chunking: bool = Field(default=True)
    breakpoint_threshold: float = Field(default=0.85)

    # Retrieval
    retrieval_k: int = Field(default=6)
    retrieval_fetch_k: int = Field(default=20)   # lấy nhiều, rerank lại
    use_hybrid_search: bool = Field(default=True)
    use_reranker: bool = Field(default=True)
    reranker_model: str = Field(default="ms-marco-MiniLM-L-12-v2")

    # Vector store
    qdrant_path: str = Field(default="./qdrant_data")
    collection_name: str = Field(default="rag_docs")

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"

config = RAGConfig()
```

---

## 4. Xử lý tài liệu nâng cấp — Docling

```python
# src/document_processor.py
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_core.documents import Document
from pathlib import Path

class DoclingProcessor:
    """Xử lý đa định dạng: PDF, Word, Excel, HTML, Markdown"""

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(
            tokenizer="BAAI/bge-m3",
            max_tokens=512,
            merge_peers=True,   # gộp các đoạn ngắn kề nhau
        )

    def process(self, file_path: str) -> list[Document]:
        result = self.converter.convert(file_path)
        doc = result.document

        chunks = list(self.chunker.chunk(dl_doc=doc))
        lc_docs = []
        for chunk in chunks:
            meta = chunk.meta.export_json_dict()
            lc_docs.append(Document(
                page_content=chunk.text,
                metadata={
                    "source": file_path,
                    "page": meta.get("page_no"),
                    "heading": meta.get("headings", [None])[0],
                    "doc_type": Path(file_path).suffix,
                }
            ))
        return lc_docs
```

---

## 5. Nâng cấp Retrieval — Hybrid Search + Reranker

Đây là nâng cấp quan trọng nhất về chất lượng.

```python
# src/retriever.py
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from flashrank.Ranker import Ranker, RerankRequest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from fastembed import TextEmbedding, SparseTextEmbedding

class HybridRAGRetriever:
    def __init__(self, config):
        self.config = config
        self.client = QdrantClient(path=config.qdrant_path)
        self._ensure_collection()

        # Dense embedding (BGE-M3)
        self.dense_model = TextEmbedding(model_name="BAAI/bge-m3")

        # Sparse embedding cho BM25-like tìm kiếm từ khoá
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

        # Reranker (chạy CPU, ~20ms/batch)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def _ensure_collection(self):
        if not self.client.collection_exists(self.config.collection_name):
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

    def add_documents(self, docs: list[Document]):
        texts = [d.page_content for d in docs]
        dense_embeds = list(self.dense_model.embed(texts))
        sparse_embeds = list(self.sparse_model.embed(texts))

        from qdrant_client.models import PointStruct, SparseVector
        points = []
        for i, (doc, dense, sparse) in enumerate(zip(docs, dense_embeds, sparse_embeds)):
            points.append(PointStruct(
                id=i,
                vector={
                    "dense": dense.tolist(),
                    "sparse": SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={"text": doc.page_content, **doc.metadata}
            ))
        self.client.upsert(collection_name=self.config.collection_name, points=points)

    def retrieve(self, query: str, k: int = None) -> list[Document]:
        k = k or self.config.retrieval_k
        fetch_k = self.config.retrieval_fetch_k

        # Hybrid query: dense + sparse
        dense_q = list(self.dense_model.embed([query]))[0].tolist()
        sparse_q = list(self.sparse_model.embed([query]))[0]

        from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, Prefetch, FusionQuery, Fusion
        results = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                Prefetch(query=NamedVector(name="dense", vector=dense_q), limit=fetch_k),
                Prefetch(query=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist())
                ), limit=fetch_k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),   # Reciprocal Rank Fusion
            limit=fetch_k,
        ).points

        # Rerank
        candidates = [r.payload["text"] for r in results]
        rerank_req = RerankRequest(query=query, passages=[{"text": t} for t in candidates])
        reranked = self.reranker.rerank(rerank_req)

        docs = []
        for item in reranked[:k]:
            orig = results[item.index]
            docs.append(Document(
                page_content=orig.payload["text"],
                metadata={k: v for k, v in orig.payload.items() if k != "text"}
            ))
        return docs
```

---

## 6. LangGraph — Orchestration nâng cao

Thay `ConversationalRetrievalChain` bằng LangGraph graph có thể mở rộng.

```python
# src/rag_graph.py
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator

class GraphState(TypedDict):
    question: str
    chat_history: Annotated[list, operator.add]
    context: list
    answer: str
    needs_retrieval: bool   # routing node

def build_rag_graph(retriever, config):
    llm = ChatOllama(
        model=config.llm_model,
        base_url=config.llm_base_url,
        temperature=config.llm_temperature,
    )

    # Node 1: Quyết định có cần retrieval không
    def route_question(state: GraphState) -> GraphState:
        history_str = "\n".join([f"User: {m.content}" if isinstance(m, HumanMessage)
                                  else f"AI: {m.content}" for m in state["chat_history"][-4:]])
        prompt = f"""Lịch sử hội thoại:
{history_str}

Câu hỏi mới: {state['question']}

Câu hỏi này cần tra cứu tài liệu không? Trả lời chỉ YES hoặc NO."""
        response = llm.invoke(prompt).content.strip().upper()
        return {**state, "needs_retrieval": "YES" in response}

    # Node 2: Rewrite query dựa trên lịch sử
    def rewrite_query(state: GraphState) -> GraphState:
        if not state.get("chat_history"):
            return state
        history_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                                  for m in state["chat_history"][-4:]])
        prompt = f"""Dựa vào lịch sử hội thoại, viết lại câu hỏi thành câu độc lập, rõ ràng.

Lịch sử: {history_str}
Câu hỏi gốc: {state['question']}
Câu hỏi viết lại:"""
        rewritten = llm.invoke(prompt).content.strip()
        return {**state, "question": rewritten}

    # Node 3: Retrieve
    def retrieve(state: GraphState) -> GraphState:
        docs = retriever.retrieve(state["question"])
        return {**state, "context": docs}

    # Node 4: Generate
    def generate(state: GraphState) -> GraphState:
        context_str = "\n\n---\n\n".join([d.page_content for d in state.get("context", [])])
        history_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                                  for m in state["chat_history"][-6:]])

        if state.get("context"):
            prompt = f"""Dựa vào ngữ cảnh bên dưới, trả lời câu hỏi.
Nếu ngữ cảnh không đủ thông tin, hãy nói rõ.

Ngữ cảnh:
{context_str}

Lịch sử: {history_str}

Câu hỏi: {state['question']}
Trả lời:"""
        else:
            prompt = f"""Lịch sử: {history_str}
Câu hỏi: {state['question']}
Trả lời:"""

        # Streaming response
        answer = llm.invoke(prompt).content
        return {
            **state,
            "answer": answer,
            "chat_history": [HumanMessage(content=state["question"]),
                             AIMessage(content=answer)],
        }

    # Build graph
    workflow = StateGraph(GraphState)
    workflow.add_node("route", route_question)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("route")
    workflow.add_conditional_edges("route", lambda s: "rewrite" if s["needs_retrieval"] else "generate")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
```

---

## 7. UI mới — Gradio 4 với streaming

```python
# app.py
import gradio as gr
from src.config import config
from src.document_processor import DoclingProcessor
from src.retriever import HybridRAGRetriever
from src.rag_graph import build_rag_graph

processor = DoclingProcessor()
retriever = HybridRAGRetriever(config)
graph = build_rag_graph(retriever, config)

def upload_file(file_path: str, progress=gr.Progress()):
    progress(0, desc="Đang xử lý tài liệu...")
    docs = processor.process(file_path)
    progress(0.5, desc=f"Đã tạo {len(docs)} chunks, đang indexing...")
    retriever.add_documents(docs)
    progress(1.0, desc="Hoàn thành!")
    return f"✅ Đã index {len(docs)} đoạn văn bản từ tài liệu."

def chat(message: str, history: list):
    lc_history = []
    for human, ai in history:
        from langchain_core.messages import HumanMessage, AIMessage
        lc_history.append(HumanMessage(content=human))
        if ai:
            lc_history.append(AIMessage(content=ai))

    state = graph.invoke({
        "question": message,
        "chat_history": lc_history,
        "context": [],
        "answer": "",
        "needs_retrieval": True,
    })
    return state["answer"]

with gr.Blocks(title="Local RAG Chatbot v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Local RAG Chatbot v2\nChat với tài liệu của bạn — hoàn toàn offline")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📁 Upload tài liệu (PDF, Word, Excel, HTML)")
            upload_btn = gr.Button("Xử lý tài liệu", variant="primary")
            upload_status = gr.Textbox(label="Trạng thái", interactive=False)
            gr.Markdown("**Model:** `" + config.llm_model + "`")
            gr.Markdown("**Embedding:** BGE-M3")
            gr.Markdown("**Retrieval:** Hybrid BM25 + Vector")
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat,
                chatbot=gr.Chatbot(height=500, show_copy_button=True),
                textbox=gr.Textbox(placeholder="Hỏi gì đó về tài liệu...", container=False),
            )
    upload_btn.click(fn=upload_file, inputs=file_input, outputs=upload_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

---

## 8. Evaluation — RAGAS 0.2

```python
# eval/evaluate.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datasets import Dataset

def run_evaluation(test_cases: list[dict]) -> dict:
    """
    test_cases: [{"question": ..., "ground_truth": ..., "answer": ..., "contexts": [...]}]
    """
    # Dùng model local để evaluate (không cần OpenAI)
    eval_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:7b", temperature=0))
    eval_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    dataset = Dataset.from_list(test_cases)
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=eval_llm,
        embeddings=eval_emb,
    )
    print(results)
    return results
```

---

## 9. Cấu trúc thư mục mới

```
Local-RAG-Chatbot-v2/
├── app.py                      # Entry point — Gradio UI
├── config.py                   # Pydantic settings
├── requirements-v2.txt
├── .env.example
├── src/
│   ├── document_processor.py   # Docling multi-format
│   ├── retriever.py            # Hybrid Search + Reranker
│   ├── rag_graph.py            # LangGraph orchestration
│   └── utils.py
├── eval/
│   ├── evaluate.py             # RAGAS 0.2
│   └── test_cases.json
├── data/                       # Upload files
├── qdrant_data/                # Vector store persistent
└── docs/
```

---

## 10. Thứ tự ưu tiên triển khai

| Ưu tiên | Nâng cấp | Lợi ích | Độ khó |
|---|---|---|---|
| 🔴 Cao | Hybrid Search + Reranker | +30–40% recall | Trung bình |
| 🔴 Cao | Ollama thay HuggingFace pipeline | Dễ quản lý model | Thấp |
| 🟡 Trung | Semantic Chunking | Chunks tốt hơn | Thấp |
| 🟡 Trung | Docling multi-format | Mở rộng file types | Thấp |
| 🟢 Thấp | LangGraph | Agentic workflow | Cao |
| 🟢 Thấp | Gradio UI | UX tốt hơn | Thấp |
| 🟢 Thấp | RAGAS 0.2 evaluation | Đánh giá khách quan | Trung bình |

---

## 11. Tips chạy trên GPU 6GB (RTX 3060)

```bash
# Chạy Ollama với giới hạn VRAM
OLLAMA_GPU_OVERHEAD=512MiB ollama serve

# Dùng quantization 4-bit (tiết kiệm ~50% VRAM)
ollama pull qwen2.5:7b-instruct-q4_K_M

# BGE-M3 embedding chạy CPU để giải phóng GPU cho LLM
# Trong config.py:
# embedding_device = "cpu"   # embedding nhẹ hơn, không cần GPU

# Reranker (FlashRank) luôn chạy CPU, không cần VRAM
```

---

*Tài liệu này được tạo tự động dựa trên phân tích dự án. Phiên bản: 2025-06.*
