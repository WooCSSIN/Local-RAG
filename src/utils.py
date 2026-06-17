"""
utils.py — Các hàm tiện ích dùng chung
"""
import logging
import io
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def render_pdf_page(file_path: str, page_number: int = 0) -> Optional[object]:
    """
    Render một trang PDF thành PIL Image.
    Args:
        file_path: Đường dẫn file PDF
        page_number: Số trang (0-indexed)
    Returns:
        PIL.Image hoặc None nếu lỗi
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image

        doc = fitz.open(file_path)
        if page_number >= len(doc):
            page_number = len(doc) - 1
        if page_number < 0:
            page_number = 0

        page = doc[page_number]
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale cho rõ nét
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        logger.error(f"Lỗi render PDF trang {page_number}: {e}")
        return None


def get_pdf_page_count(file_path: str) -> int:
    """Trả về số trang của PDF."""
    try:
        import fitz
        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def format_sources(docs: list) -> str:
    """
    Format danh sách source documents thành chuỗi dễ đọc.
    """
    if not docs:
        return ""
    seen = set()
    lines = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        key = f"{filename}-{page}"
        if key not in seen:
            seen.add(key)
            if page is not None:
                lines.append(f"📄 {filename}, trang {page + 1}")
            else:
                lines.append(f"📄 {filename}")
    return "\n".join(lines)


def check_ollama_connection(base_url: str = "http://localhost:11434") -> tuple[bool, str]:
    """
    Kiểm tra Ollama có đang chạy không.
    Returns:
        (is_running: bool, message: str)
    """
    try:
        import urllib.request
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        import json
        data = json.loads(req.read())
        models = [m["name"] for m in data.get("models", [])]
        return True, f"✅ Ollama đang chạy. Models: {', '.join(models) or 'chưa có model nào'}"
    except Exception as e:
        return False, f"❌ Ollama chưa chạy hoặc lỗi kết nối: {e}\nChạy lệnh: ollama serve"


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Lấy danh sách models đang có trong Ollama."""
    try:
        import urllib.request
        import json
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        data = json.loads(req.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Cắt ngắn text để hiển thị."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def setup_logging(level: str = "INFO"):
    """Cấu hình logging cho toàn bộ app."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
