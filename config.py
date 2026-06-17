"""
config.py — Cấu hình toàn bộ dự án dùng Pydantic Settings
Thay thế config.yaml cũ, hỗ trợ .env và type checking.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class RAGConfig(BaseSettings):
# LLM backend
    llm_provider: str = Field(
        default="groq",
        description="LLM provider: groq | ollama | openai"
    )
    # Groq (miễn phí, không cần cài gì)
    groq_api_key: str = Field(default="", description="Groq API key — lấy tại console.groq.com")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")

    # Ollama (local, cần cài ollama)
    llm_model: str = Field(default="qwen2.5:7b", description="Tên model Ollama")
    llm_base_url: str = Field(default="http://localhost:11434", description="URL Ollama server")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, gt=0)

    # --------------------------------------------------
    # Embedding
    # --------------------------------------------------
    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="Model FastEmbed. Options: nomic-ai/nomic-embed-text-v1.5, BAAI/bge-large-en-v1.5, BAAI/bge-small-en-v1.5"
    )
    embedding_device: str = Field(
        default="cpu",
        description="cpu hoặc cuda. Dùng cpu để giải phóng GPU cho LLM"
    )

    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    use_semantic_chunking: bool = Field(
        default=False,
        description="True = semantic chunking (cần thêm RAM), False = recursive character splitter"
    )
    breakpoint_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    retrieval_k: int = Field(default=6, gt=0, description="Số docs trả về cuối cùng sau rerank")
    retrieval_fetch_k: int = Field(default=20, gt=0, description="Số docs fetch trước khi rerank")
    use_hybrid_search: bool = Field(default=True, description="Kết hợp BM25 + vector search")
    use_reranker: bool = Field(default=True, description="Rerank bằng FlashRank cross-encoder")
    reranker_model: str = Field(default="ms-marco-MiniLM-L-12-v2")

    # --------------------------------------------------
    # Vector store — Qdrant
    # --------------------------------------------------
    qdrant_path: str = Field(default="./qdrant_data", description="Đường dẫn lưu vector store local")
    collection_name: str = Field(default="rag_docs")

    # --------------------------------------------------
    # UI
    # --------------------------------------------------
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=7860)
    max_chat_history: int = Field(default=6, description="Số lượt hội thoại giữ trong context")

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"
        extra = "ignore"


# Singleton instance — import ở bất kỳ đâu
config = RAGConfig()


if __name__ == "__main__":
    print("=== RAG Config ===")
    for field, value in config.model_dump().items():
        print(f"  {field}: {value}")
