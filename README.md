# 🤖 Local RAG Chatbot v2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.3-green?logo=langchain)
![Gradio](https://img.shields.io/badge/Gradio-6.x-orange?logo=gradio)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)

**Chat với tài liệu của bạn — thông minh, nhanh, bảo mật**

[Demo](#-demo) • [Tính năng](#-tính-năng) • [Cài đặt](#-cài-đặt-nhanh) • [Cấu hình](#-cấu-hình) • [Kiến trúc](#-kiến-trúc)

</div>

---

## 📌 Giới thiệu

**Local RAG Chatbot v2** là hệ thống chat với tài liệu sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)**. Upload PDF, Word, TXT... và đặt câu hỏi bằng ngôn ngữ tự nhiên — hệ thống tự tìm đúng đoạn liên quan và trả lời chính xác kèm nguồn.


---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 📄 **Multi-format** | Hỗ trợ PDF, TXT, HTML, Markdown |
| 🔍 **Hybrid Search** | Kết hợp BM25 (keyword) + FAISS (semantic) |
| 🎯 **Reranker** | FlashRank cross-encoder tái xếp hạng kết quả |
| 🧠 **LangGraph** | Pipeline thông minh: route → rewrite → retrieve → generate |
| 💬 **Streaming** | Trả lời realtime token-by-token |
| 🔒 **Bảo mật** | Tài liệu lưu local, không gửi ra ngoài |
| 🌐 **Multi LLM** | Hỗ trợ Groq (miễn phí), Ollama (local), OpenAI |
| 📊 **Evaluation** | RAGAS metrics với Ollama local |

---

## 🆚 So sánh v1 vs v2

| Thành phần | v1 | v2 |
|---|---|---|
| **LLM** | HuggingFace pipeline (Llama2 GPTQ) | Groq / Ollama / OpenAI |
| **Embedding** | MiniLM-L6 (384 dim) | nomic-embed-text (768 dim) |
| **Vector store** | ChromaDB in-memory | FAISS persistent |
| **Retrieval** | Pure vector | Hybrid BM25 + Vector + Reranker |
| **Reranker** | Cohere API (có phí) | FlashRank (local, miễn phí) |
| **Orchestration** | ConversationalRetrievalChain | LangGraph stateful graph |
| **UI** | Streamlit basic | Gradio 6 streaming |
| **Config** | YAML cứng | Pydantic Settings + `.env` |

---

## 🚀 Cài đặt nhanh

### Yêu cầu
- Python 3.10+
- 4GB RAM (tối thiểu)
- Groq API key miễn phí (hoặc Ollama)

### Bước 1 — Clone & cài dependencies

```bash
git clone https://github.com/WooCSSIN/Local-RAG.git
cd Local-RAG
pip install -r requirements-v2.txt
```

### Bước 2 — Cấu hình

```bash
cp .env.example .env
```

Mở file `.env` và điền thông tin:

```env
# Dùng Groq (miễn phí, không cần cài gì)
RAG_LLM_PROVIDER=groq
RAG_GROQ_API_KEY=gsk_...  # Lấy tại https://console.groq.com
RAG_GROQ_MODEL=llama-3.3-70b-versatile
```

> **Lấy Groq API key miễn phí:** Đăng ký tại [console.groq.com](https://console.groq.com) → API Keys → Create

### Bước 3 — Chạy app

```bash
python app.py
```

Mở trình duyệt: **http://localhost:7860**

---

## ⚙️ Cấu hình

Tất cả cấu hình trong file `.env` (copy từ `.env.example`):

```env
# LLM Provider
RAG_LLM_PROVIDER=groq          # groq | ollama | openai

# Groq
RAG_GROQ_API_KEY=gsk_...
RAG_GROQ_MODEL=llama-3.3-70b-versatile

# Ollama (nếu dùng local)
RAG_LLM_MODEL=qwen2.5:7b
RAG_LLM_BASE_URL=http://localhost:11434

# Embedding
RAG_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Retrieval
RAG_RETRIEVAL_K=6              # Số docs trả về
RAG_USE_HYBRID_SEARCH=true     # BM25 + Vector
RAG_USE_RERANKER=true          # FlashRank reranker

# Chunking
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=64
```

### Dùng Ollama (local, không cần internet)

```bash
# Cài Ollama
# Windows: https://ollama.com/download/windows
# Linux:   curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen2.5:7b      # 5GB VRAM
ollama pull llama3.2:3b     # 2GB VRAM (nhẹ)

# Chạy server
ollama serve
```

Đổi trong `.env`:
```env
RAG_LLM_PROVIDER=ollama
RAG_LLM_MODEL=qwen2.5:7b
```

---

## 🏗️ Kiến trúc

```
┌─────────────────────────────────────────────────┐
│                  Gradio 6 UI                     │
│         Upload │ Chat │ PDF Preview              │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │   LangGraph     │
        │  ┌───────────┐  │
        │  │   Route   │  │  ← Có cần retrieval?
        │  └─────┬─────┘  │
        │  ┌─────▼─────┐  │
        │  │  Rewrite  │  │  ← Viết lại query
        │  └─────┬─────┘  │
        │  ┌─────▼─────┐  │
        │  │ Retrieve  │  │  ← Hybrid search
        │  └─────┬─────┘  │
        │  ┌─────▼─────┐  │
        │  │ Generate  │  │  ← Stream LLM
        │  └───────────┘  │
        └────────┬────────┘
                 │
    ┌────────────▼────────────┐
    │     Hybrid Retriever    │
    │  BM25 ──┐               │
    │         ├── RRF Fusion  │
    │  FAISS ─┘       │       │
    │              Reranker   │
    │           (FlashRank)   │
    └─────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │    Document Processor   │
    │  PDF │ TXT │ HTML │ MD  │
    │    Chunking + Embed     │
    └─────────────────────────┘
```

---

## 📁 Cấu trúc dự án

```
Local-RAG/
├── app.py                    # 🚀 Entry point — Gradio UI
├── config.py                 # ⚙️  Pydantic Settings
├── requirements-v2.txt       # 📦 Dependencies
├── .env.example              # 🔑 Template cấu hình
├── setup_v2.py               # 🔧 Health check script
│
├── src/
│   ├── document_processor.py # 📄 Xử lý đa định dạng
│   ├── retriever.py          # 🔍 Hybrid BM25+FAISS+Reranker
│   ├── rag_graph.py          # 🕸️  LangGraph pipeline
│   ├── llm_factory.py        # 🤖 Groq/Ollama/OpenAI factory
│   └── utils.py              # 🛠️  Tiện ích
│
├── eval/
│   ├── evaluate.py           # 📊 RAGAS evaluation
│   └── ...                   # Notebooks gốc
│
└── qdrant_data/              # 💾 Vector store (local)
```

---

## 📊 Evaluation

Đánh giá chất lượng RAG với RAGAS metrics (không cần OpenAI):

```bash
# Tạo test cases
python -c "
from eval.evaluate import generate_test_cases_from_rag
# ... xem eval/evaluate.py để biết thêm
"

# Chạy evaluation
python eval/evaluate.py --input eval/test_cases.json --output eval/metrics/results.csv
```

Metrics được đo: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`

---

## 🔒 Bảo mật

- ✅ `.env` được thêm vào `.gitignore` — API key không bao giờ bị push
- ✅ Tài liệu lưu local, không gửi ra ngoài
- ✅ Chỉ câu hỏi và context được gửi đến Groq/Ollama API
- ⚠️ Không commit file `.env` lên GitHub
- ⚠️ Nếu lỡ push key → xóa và tạo key mới ngay tại [console.groq.com](https://console.groq.com)

---

## 🛠️ Kiểm tra môi trường

```bash
python setup_v2.py
```

Output mẫu:
```
==================================================
  Local RAG Chatbot v2 — Setup Check
==================================================
Python 3.14.4  ✅
Ollama   : ✅ ollama version 0.x.x
  14/14 packages OK
Index    : ✅ 301 docs trong vector store
```

---

## 📝 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq (Llama 3.3 70B) / Ollama / OpenAI |
| **Embedding** | nomic-embed-text-v1.5 (FastEmbed) |
| **Vector Store** | FAISS (persistent) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranker** | FlashRank (ms-marco-MiniLM) |
| **Orchestration** | LangGraph 1.x |
| **Framework** | LangChain 1.x |
| **UI** | Gradio 6.x |
| **Config** | Pydantic Settings |
| **Evaluation** | RAGAS 0.4 |

---

## 🤝 Contributing

Pull requests welcome! Các hướng cải thiện tiếp theo:

- [ ] Docling integration (Word, Excel, PowerPoint)
- [ ] Multi-session support
- [ ] Chat history persistent (SQLite)
- [ ] Docker compose deployment
- [ ] Vietnamese embedding model

---

## 📄 License

Apache 2.0 — Dựa trên [haooyuee/Local-RAG-Chatbot](https://github.com/haooyuee/Local-RAG-Chatbot)

---

<div align="center">
Made with ❤️ | <a href="https://github.com/WooCSSIN">WooCSSIN</a>
</div>
