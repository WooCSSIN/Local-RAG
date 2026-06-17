"""
llm_factory.py — Tạo LLM instance theo provider được cấu hình
Hỗ trợ: groq (miễn phí), ollama (local), openai
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_llm(config) -> Any:
    """
    Trả về LLM instance dựa trên config.llm_provider.
    
    Ưu tiên:
      1. Nếu GROQ_API_KEY có trong env/config → dùng Groq
      2. Nếu Ollama đang chạy → dùng Ollama  
      3. Fallback: raise lỗi rõ ràng
    """
    provider = getattr(config, "llm_provider", "auto").lower()

    # Auto-detect nếu không set
    if provider == "auto":
        provider = _auto_detect(config)

    if provider == "groq":
        return _make_groq(config)
    elif provider == "ollama":
        return _make_ollama(config)
    elif provider == "openai":
        return _make_openai(config)
    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Dùng: groq | ollama | openai"
        )


def _auto_detect(config) -> str:
    """Tự động chọn provider có sẵn."""
    # Thử Groq trước
    groq_key = getattr(config, "groq_api_key", "") or ""
    if groq_key and groq_key != "gsk_xxxxxxxxxxxxxxxxxxxx":
        logger.info("Auto-detect: dùng Groq")
        return "groq"

    # Thử Ollama
    try:
        import urllib.request
        urllib.request.urlopen(
            f"{getattr(config, 'llm_base_url', 'http://localhost:11434')}/api/tags",
            timeout=2
        )
        logger.info("Auto-detect: dùng Ollama")
        return "ollama"
    except Exception:
        pass

    raise RuntimeError(
        "Không tìm thấy LLM backend nào!\n\n"
        "Chọn một trong hai:\n"
        "  Option A — Groq (miễn phí, không cần cài):\n"
        "    1. Đăng ký tại https://console.groq.com\n"
        "    2. Thêm vào .env: RAG_GROQ_API_KEY=gsk_...\n"
        "                      RAG_LLM_PROVIDER=groq\n\n"
        "  Option B — Ollama (local):\n"
        "    1. Tải: https://ollama.com/download/windows\n"
        "    2. Chạy: ollama pull llama3.2:3b\n"
        "    3. Chạy: ollama serve\n"
    )


def _make_groq(config) -> Any:
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Chạy: pip install langchain-groq")

    api_key = getattr(config, "groq_api_key", "") or ""
    if not api_key:
        import os
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Groq API key chưa được set.\n"
            "Thêm vào .env: RAG_GROQ_API_KEY=gsk_...\n"
            "Lấy key miễn phí: https://console.groq.com"
        )

    model = getattr(config, "groq_model", "llama-3.3-70b-versatile")
    temp = getattr(config, "llm_temperature", 0.1)
    max_tokens = getattr(config, "llm_max_tokens", 2048)

    logger.info(f"LLM: Groq / {model}")
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temp,
        max_tokens=max_tokens,
    )


def _make_ollama(config) -> Any:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("Chạy: pip install langchain-ollama")

    model = getattr(config, "llm_model", "qwen2.5:7b")
    base_url = getattr(config, "llm_base_url", "http://localhost:11434")
    temp = getattr(config, "llm_temperature", 0.1)
    max_tokens = getattr(config, "llm_max_tokens", 2048)

    logger.info(f"LLM: Ollama / {model} @ {base_url}")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temp,
        num_predict=max_tokens,
    )


def _make_openai(config) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Chạy: pip install langchain-openai")

    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")
    logger.info("LLM: OpenAI / gpt-4o-mini")
    return ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=getattr(config, "llm_temperature", 0.1),
    )


def get_provider_name(config) -> str:
    """Trả về tên provider dễ đọc để hiển thị trên UI."""
    provider = getattr(config, "llm_provider", "auto").lower()
    if provider == "groq":
        return f"Groq / {getattr(config, 'groq_model', 'llama-3.3-70b-versatile')}"
    elif provider == "ollama":
        return f"Ollama / {getattr(config, 'llm_model', 'qwen2.5:7b')}"
    elif provider == "openai":
        return "OpenAI / gpt-4o-mini"
    return "Auto-detect"
