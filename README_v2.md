# 🤖 Local RAG Chatbot v2

Nâng cấp hoàn toàn từ [haooyuee/Local-RAG-Chatbot](https://github.com/haooyuee/Local-RAG-Chatbot).  
Chat với tài liệu của bạn — **hoàn toàn offline**, không gửi dữ liệu ra ngoài.

---

## ✨ Điểm khác biệt so với v1

| Thành phần | v1 (cũ) | v2 (mới) |
|---|---|---|
| **LLM** | HuggingFace pipeline (Llama2 GPTQ) | **Ollama** (Llama3, Qwen2.5, Phi3.5...) |
| **Embedding** | MiniLM-L6 (384 chiều) | **BGE-M3** (1024 chiều, đa ngôn ngữ) |
| **Retrieval** | Pure vector (Chroma) | **Hybrid BM25 + Vector + Reranker** |
| **Reranker** | Cohere API (cần internet) | **FlashRank** (local CPU, ~20ms) |
| **Orchestration** | `ConversationalRetrievalChain` | **LangGraph** (stateful graph) |
| **Document types** | PDF only | PDF, Word, Excel, HTML, Markdown, TXT |
| **UI** | Streamlit basic | **Gradio 4** (multi-file, responsive) |
| **Config** | YAML cứng nhắc | **Pydantic Settings** + `.env` |
| **Evaluation** | OpenAI required | **RAGAS 0.2 + Ollama** (hoàn toàn local) |

---

## 🚀 Cài đặt nhanh

### Bước 1 — Cài Ollama

**Windows** (PowerShell):
```powershell
irm https://ollama.com/install.ps1 | iex
```
Hoặc tải tại: https://ollama.com/download/windows

### Bước 2 — Pull LLM model

```bash
# Chọn theo VRAM
ollama pull qwen2.5:7b        # 5GB VRAM — khuyên dùng
ollama pull llama3.2:3b       # 2GB VRAM — nhẹ nhất
ollama pull phi4-mini         # 3.8GB VRAM — nhanh
```

### Bước 3 — Tạo môi trường Python

```bash
# Python 3.10+ (đã test trên 3.14)
pip install -r requirements-v2.txt
```

### Bước 4 — Chạy kiểm tra

```bash
# Khởi động Ollama server
ollama serve

# Kiểm tra môi trường
python setup_v2.py

# Test pipeline (sau khi ollama serve)
python test_ollama.py
```

### Bước 5 — Chạy app

```bash
python app.py
# Mở: http://localhost:7860
```

---

## 🛠️ Kiểm tra môi trường

```bash
python setup_v2.py
```

Script này sẽ kiểm tra Python, CUDA, Ollama, và tất cả packages cần thiết.

---

## 📁 Cấu trúc dự án

```
Local-RAG-v2/
├── app.py                      # 🚀 Entry point (Gradio UI)
├── config.py                   # ⚙️ Pydantic Settings
├── requirements-v2.txt         # 📦 Dependencies mới
├── setup_v2.py                 # 🔧 Setup & health check
├── .env.example                # 🔑 Template biến môi trường
│
├── src/
│   ├── document_processor.py   # 📄 Docling multi-format processor
│   ├── retriever.py            # 🔍 Hybrid Search + Reranker
│   ├── rag_graph.py            # 🕸️ LangGraph orchestration
│   └── utils.py                # 🛠️ Tiện ích chung
│
├── eval/
│   ├── evaluate.py             # 📊 RAGAS 0.2 evaluation
│   └── (notebooks cũ giữ lại)
│
├── qdrant_data/                # 💾 Vector store (tự tạo)
└── docs/
```

---

## 💡 Sử dụng

1. **Upload tài liệu**: Kéo thả PDF, Word, Excel, HTML, TXT vào vùng upload
2. **Đặt câu hỏi**: Nhập câu hỏi và nhấn Enter hoặc nút Gửi
3. **Xem nguồn**: Mỗi câu trả lời kèm nguồn tài liệu và số trang

---

## 📊 Evaluation

Chạy evaluation với Ollama (không cần OpenAI):

```bash
# Tạo test cases JSON trước, sau đó:
python eval/evaluate.py --input eval/test_cases.json --output eval/metrics/results.csv
```

Format test cases:
```json
[
  {
    "question": "Câu hỏi?",
    "answer": "Câu trả lời của model",
    "contexts": ["chunk 1", "chunk 2"],
    "ground_truth": "Đáp án đúng"
  }
]
```

---

## ⚡ Tips cho GPU 6GB (RTX 3060)

```bash
# Chạy Ollama tiết kiệm VRAM
OLLAMA_GPU_OVERHEAD=512MiB ollama serve

# Dùng quantization 4-bit
ollama pull qwen2.5:7b-instruct-q4_K_M

# Trong .env: đặt embedding chạy CPU để giải phóng GPU cho LLM
RAG_EMBEDDING_DEVICE=cpu
```

---

## 🔧 Cấu hình nâng cao

Chỉnh file `.env` hoặc `config.py`:

| Biến | Mặc định | Mô tả |
|---|---|---|
| `RAG_LLM_MODEL` | `qwen2.5:7b` | Model Ollama |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Model embedding |
| `RAG_RETRIEVAL_K` | `6` | Số docs trả về |
| `RAG_USE_HYBRID_SEARCH` | `true` | BM25 + vector |
| `RAG_USE_RERANKER` | `true` | FlashRank reranker |
| `RAG_CHUNK_SIZE` | `512` | Kích thước chunk |

---

*Dựa trên: haooyuee/Local-RAG-Chatbot | Nâng cấp: 2025-06*
