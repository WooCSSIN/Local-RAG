---
title: Local RAG Chatbot Gree
emoji: "\U0001F916"
colorFrom: blue
colorTo: purple
license: apache-2.0
short_description: Agentic RAG Chatbot — Chat with your documents using Hybrid Search + LangGraph
---

# Local RAG Chatbot Gree

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

**Local RAG Chatbot Gree** là hệ thống chat với tài liệu sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)**. Upload PDF, Word, TXT... và đặt câu hỏi bằng ngôn ngữ tự nhiên — hệ thống tự tìm đúng đoạn liên quan và trả lời chính xác kèm nguồn.


---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 📄 **Multi-format** | Hỗ trợ PDF, DOCX, XLSX, TXT, HTML, Markdown, ảnh OCR |
| 🔍 **Hybrid Search** | Kết hợp BM25 (keyword) + FAISS (semantic) + RRF Fusion |
| 🎯 **Reranker** | FlashRank cross-encoder tái xếp hạng kết quả |
| 🧠 **Agentic LangGraph** | 9-node pipeline: route → decompose → rewrite → retrieve → assess → tool → generate → grade |
| 🤖 **Self-Reflection** | Tự đánh giá chất lượng câu trả lời, phát hiện hallucination |
| 🔧 **Query Decomposition** | Tách câu hỏi phức tạp thành sub-queries |
| 🌐 **Web Search** | Tích hợp DuckDuckGo khi tài liệu local không đủ |
| 💬 **Streaming** | Trả lời realtime token-by-token |
| 🔒 **Bảo mật** | Tài liệu lưu local, không gửi ra ngoài |
| 🌐 **Multi LLM** | Hỗ trợ Groq (miễn phí), Ollama (local), OpenAI |
| 📊 **Evaluation** | RAGAS metrics + auto-benchmark |

---

## 🆚 So sánh v1 vs Gree

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

## Deploy lên Render.com (Miễn phí)

### Bước 1 — Push code lên GitHub

```bash
git add -A
git commit -m "feat: ready for Render deployment"
git push origin main
```

### Bước 2 — Tạo Web Service trên Render

1. Đăng nhập [render.com](https://render.com) (bằng GitHub account)
2. **New +** → **Web Service** → Chọn repo `Local-RAG`
3. Cấu hình:
   - **Name**: `local-rag-chatbot`
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: `Free`
4. Thêm **Environment Variables** (mục Environment):
   - `RAG_LLM_PROVIDER` = `groq`
   - `RAG_GROQ_API_KEY` = `gsk_...` (lấy tại [console.groq.com](https://console.groq.com))
   - `RAG_USE_HYBRID_SEARCH` = `true`
   - `RAG_USE_RERANKER` = `true`
5. **Create Web Service** → Render tự động build & deploy

> URL sẽ là: `https://local-rag-chatbot.onrender.com`

### Lưu ý (Free tier)

- Cold start ~30s sau khi không dùng 15 phút
- Index trống khi khởi động — upload tài liệu qua UI
- Embedding model (~270MB) tự download lần đầu

---

## 📦 Đóng gói thành phần mềm Windows (.exe)

### Yêu cầu

- Windows 10/11
- Python 3.11
- ~10GB disk space để build
- Cài [Inno Setup](https://jrsoftware.org/isinfo.php) (nếu muốn tạo installer)

### Bước 1 — Cài PyInstaller

```bash
pip install pyinstaller
```

### Bước 2 — Build

```bash
python build_exe.py
```

Output: `D:\Local-RAG-Build\dist\Local-RAG\`

### Bước 3 — Tạo installer (tùy chọn)

Mở [installer.iss](installer.iss) bằng Inno Setup → Compile.

Output: `D:\Local-RAG-Build\installer\Local-RAG-Chatbot-Setup.exe`

### Lưu ý

- Lần đầu chạy app, embedding model (~270MB) sẽ tự động download
- Nên tạo file `.env` trong folder app với `RAG_GROQ_API_KEY`
- File `.exe` có thể bị Windows Defender false positive (do PyInstaller) — add exception nếu cần

---

## Cài đặt Local

### Yêu cầu
- Python 3.10+
- 4GB RAM (tối thiểu)
- Groq API key miễn phí (hoặc Ollama)

### Bước 1 — Clone & cài dependencies

```bash
git clone https://github.com/WooCSSIN/Local-RAG.git
cd Local-RAG
pip install -r requirements.txt
```

> `requirements-v2.txt` là phiên bản pinned đầy đủ (có thể dùng thay thế).

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
├── requirements.txt          # 📦 Dependencies
├── requirements-v2.txt       # 📌 Pinned dependencies (fallback)
├── .env.example              # 🔑 Template cấu hình
├── setup_v2.py               # 🔧 Health check script
│
├── src/
│   ├── agentic_graph.py      # 🧠 Agentic LangGraph (9-node pipeline)
│   ├── agents/               # 🤖 Specialized agents
│   │   ├── decomposer.py     #    Query decomposition
│   │   ├── grader.py         #    Answer grading / self-reflection
│   │   ├── retrieval_agent.py#    Multi-step retrieval
│   │   └── tools.py          #    Web search (DuckDuckGo)
│   ├── prompts.py            # 📝 Centralized prompt templates
│   ├── memory.py             # 💭 Conversation summarization
│   ├── document_processor.py # 📄 Multi-format + adaptive chunking
│   ├── retriever.py          # 🔍 Hybrid BM25+FAISS+Reranker (O(1) RRF)
│   ├── rag_graph.py          # 🕸️  Base LangGraph (backward compat)
│   ├── llm_factory.py        # 🤖 Groq/Ollama/OpenAI factory
│   ├── session_manager.py    # 💾 Persistent sessions
│   └── utils.py              # 🛠️  Utilities
│
├── eval/
│   ├── evaluate.py           # 📊 RAGAS evaluation
│   └── auto_benchmark.py     # 📈 Auto-benchmark tool
│
└── qdrant_data/              # 💾 Vector store (local, gitignored)
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

- [x] Agentic RAG với multi-agent LangGraph
- [x] Query decomposition cho câu hỏi phức tạp
- [x] Self-reflection / answer grading
- [x] Web search integration (DuckDuckGo)
- [x] Adaptive chunking (heading/table/code-aware)
- [x] Conversation memory summarization
- [x] Auto-benchmark tool
- [ ] Multi-user authentication
- [ ] Per-project collections
- [ ] SQLite persistent storage
- [ ] Docker compose deployment
- [ ] Async I/O optimization

---

## 📄 License

Apache 2.0 — Dựa trên [haooyuee/Local-RAG-Chatbot](https://github.com/haooyuee/Local-RAG-Chatbot)

---

<div align="center">
Made with ❤️ | <a href="https://github.com/WooCSSIN">WooCSSIN</a>
</div>
