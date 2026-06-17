"""
document_processor.py — Xử lý tài liệu đa định dạng
Hỗ trợ: PDF, Word (.docx), Excel (.xlsx), HTML, Markdown, TXT
Dùng Docling với fallback PyPDF nếu Docling lỗi.
"""
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Các định dạng Docling hỗ trợ
DOCLING_SUPPORTED = {".pdf", ".docx", ".xlsx", ".html", ".htm", ".md", ".markdown"}
# Fallback đơn giản
TEXT_FORMATS = {".txt"}


class DocumentProcessor:
    """
    Xử lý tài liệu đa định dạng, trả về list[Document] sẵn để index.
    Ưu tiên Docling cho chất lượng tốt nhất. Fallback PyPDF/text nếu cần.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._docling_available = self._check_docling()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "(?<=[.?!])", " ", ""],
        )

    def _check_docling(self) -> bool:
        """Kiểm tra Docling đã cài chưa."""
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            return True
        except ImportError:
            logger.warning("Docling chưa được cài. Dùng PyPDF fallback cho PDF.")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str) -> list[Document]:
        """
        Xử lý file và trả về list Document đã chunk.
        Args:
            file_path: Đường dẫn file (str hoặc Path)
        Returns:
            list[Document] sẵn để add vào vector store
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File không tồn tại: {file_path}")

        suffix = path.suffix.lower()
        logger.info(f"Xử lý file: {path.name} (loại: {suffix})")

        if suffix in DOCLING_SUPPORTED and self._docling_available:
            docs = self._process_with_docling(str(path))
        elif suffix == ".pdf":
            docs = self._process_pdf_fallback(str(path))
        elif suffix in TEXT_FORMATS:
            docs = self._process_text(str(path))
        else:
            raise ValueError(f"Định dạng chưa hỗ trợ: {suffix}. "
                             f"Hỗ trợ: {DOCLING_SUPPORTED | TEXT_FORMATS}")

        logger.info(f"Hoàn thành: {len(docs)} chunks từ {path.name}")
        return docs

    def process_multiple(self, file_paths: list[str]) -> list[Document]:
        """Xử lý nhiều file cùng lúc."""
        all_docs = []
        for fp in file_paths:
            try:
                docs = self.process(fp)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Lỗi xử lý {fp}: {e}")
        return all_docs

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _process_with_docling(self, file_path: str) -> list[Document]:
        """Dùng Docling — chất lượng cao nhất, hiểu cấu trúc document."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        dl_doc = result.document

        # Thử dùng HybridChunker nếu có
        try:
            from docling.chunking import HybridChunker
            chunker = HybridChunker(max_tokens=self.chunk_size, merge_peers=True)
            chunks = list(chunker.chunk(dl_doc=dl_doc))
            docs = []
            for chunk in chunks:
                meta = {}
                try:
                    meta = chunk.meta.export_json_dict()
                except Exception:
                    pass
                docs.append(Document(
                    page_content=chunk.text,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "page": meta.get("page_no"),
                        "heading": (meta.get("headings") or [None])[0],
                        "doc_type": Path(file_path).suffix,
                        "processor": "docling-hybrid",
                    }
                ))
            return docs
        except Exception:
            # Fallback: export markdown rồi split
            md_text = dl_doc.export_to_markdown()
            return self._split_text(md_text, file_path, processor="docling-md")

    def _process_pdf_fallback(self, file_path: str) -> list[Document]:
        """Fallback dùng PyPDF khi Docling không có."""
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        all_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            all_text += f"\n\n[Trang {i+1}]\n{text}"
        return self._split_text(all_text, file_path, processor="pypdf")

    def _process_text(self, file_path: str) -> list[Document]:
        """Xử lý file text thuần."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return self._split_text(text, file_path, processor="text")

    def _split_text(self, text: str, source: str, processor: str = "default") -> list[Document]:
        """Tạo LangChain Document rồi split."""
        base_doc = Document(
            page_content=text,
            metadata={
                "source": source,
                "filename": Path(source).name,
                "doc_type": Path(source).suffix,
                "processor": processor,
            }
        )
        return self._splitter.split_documents([base_doc])


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <file_path>")
        sys.exit(1)
    proc = DocumentProcessor(chunk_size=512, chunk_overlap=64)
    docs = proc.process(sys.argv[1])
    print(f"\n✅ {len(docs)} chunks")
    print(f"Chunk đầu tiên:\n{docs[0].page_content[:300]}")
    print(f"Metadata: {docs[0].metadata}")
