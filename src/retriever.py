"""
retriever.py — Hybrid Search + Reranker
Kết hợp BM25 (keyword) + Dense vector (semantic) + FlashRank reranker.
R1: FAISS index persistent (lưu disk, load < 1s)
R4: Duplicate guard (SHA-256 hash)
"""
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HybridRAGRetriever:
    """
    Retriever nâng cao với 3 chiến lược:
    1. Dense search (FAISS + FastEmbed BGE-M3)
    2. BM25 keyword search
    3. FlashRank cross-encoder reranker

    Dùng Reciprocal Rank Fusion (RRF) để kết hợp kết quả.
    """

    def __init__(self, config):
        self.config = config
        self.documents: list[Document] = []  # lưu toàn bộ docs để BM25

        # Lazy-loaded components
        self._faiss_index = None
        self._bm25 = None
        self._embed_model = None
        self._reranker = None

        # Đường dẫn lưu trữ persistent
        self._store_dir = Path(config.qdrant_path)  # reuse path config
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._faiss_bin_path = self._store_dir / "faiss_index.bin"   # R1: lưu FAISS binary
        self._docs_path = self._store_dir / "documents.pkl"
        self._hashes_path = self._store_dir / "file_hashes.pkl"      # R4: duplicate guard
        self._file_hashes: set[str] = set()

        # Load nếu đã có data
        self._load_persistent()

    # ------------------------------------------------------------------
    # Embedding model (FastEmbed — chạy local, không cần API)
    # ------------------------------------------------------------------

    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from fastembed import TextEmbedding
                logger.info(f"Tải embedding model: {self.config.embedding_model}")
                self._embed_model = TextEmbedding(
                    model_name=self.config.embedding_model,
                    cache_dir=str(self._store_dir / "embed_cache"),
                )
                logger.info("✅ Embedding model sẵn sàng")
            except ImportError:
                logger.warning("fastembed chưa cài. Dùng HuggingFaceEmbeddings fallback.")
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embed_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        return self._embed_model

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed danh sách text, hỗ trợ cả FastEmbed và HuggingFaceEmbeddings."""
        model = self._get_embed_model()
        try:
            # FastEmbed interface — nomic cần prefix "passage: "
            model_name = self.config.embedding_model
            if "nomic" in model_name:
                texts = [f"passage: {t}" for t in texts]
            embeddings = list(model.embed(texts))
            return [e.tolist() for e in embeddings]
        except AttributeError:
            # LangChain interface
            return model.embed_documents(texts)

    def _embed_query(self, query: str) -> list[float]:
        """Embed một câu query."""
        model = self._get_embed_model()
        try:
            # nomic cần prefix "query: " cho query
            model_name = self.config.embedding_model
            if "nomic" in model_name:
                query = f"query: {query}"
            result = list(model.embed([query]))
            return result[0].tolist()
        except AttributeError:
            return model.embed_query(query)

    # ------------------------------------------------------------------
    # FAISS vector store
    # ------------------------------------------------------------------

    def _get_faiss(self):
        if self._faiss_index is None and self.documents:
            self._build_faiss()
        return self._faiss_index

    def _build_faiss(self):
        """Xây dựng FAISS index từ documents hiện có."""
        import faiss
        import numpy as np

        if not self.documents:
            return

        logger.info(f"Building FAISS index cho {len(self.documents)} docs...")
        texts = [d.page_content for d in self.documents]
        embeddings = self._embed_texts(texts)
        embeddings_np = np.array(embeddings, dtype="float32")

        # Normalize cho cosine similarity
        faiss.normalize_L2(embeddings_np)
        dim = embeddings_np.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sau normalize
        self._faiss_index.add(embeddings_np)

        # R1: Lưu FAISS index xuống disk ngay sau khi build
        self._save_faiss_index()
        logger.info("✅ FAISS index ready")

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _get_bm25(self):
        if self._bm25 is None and self.documents:
            self._build_bm25()
        return self._bm25

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        texts = [d.page_content.lower().split() for d in self.documents]
        self._bm25 = BM25Okapi(texts)
        logger.info("✅ BM25 index ready")

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from flashrank import Ranker
                self._reranker = Ranker(model_name=self.config.reranker_model)
                logger.info(f"✅ Reranker ready: {self.config.reranker_model}")
            except Exception as e:
                logger.warning(f"Reranker không khả dụng: {e}. Bỏ qua reranking.")
        return self._reranker

    # ------------------------------------------------------------------
    # R4: Duplicate guard
    # ------------------------------------------------------------------

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Tính SHA-256 hash của file để phát hiện duplicate."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_duplicate(self, file_path: str) -> bool:
        """Trả về True nếu file đã được index (dựa trên SHA-256 hash)."""
        file_hash = self.compute_file_hash(file_path)
        return file_hash in self._file_hashes

    def register_file(self, file_path: str):
        """Đăng ký hash của file sau khi index thành công."""
        file_hash = self.compute_file_hash(file_path)
        self._file_hashes.add(file_hash)
        self._save_hashes()

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[Document]):
        """Thêm documents mới vào index."""
        if not docs:
            return
        self.documents.extend(docs)
        self._faiss_index = None
        self._bm25 = None
        self._build_faiss()
        self._build_bm25()
        self._save_persistent()
        logger.info(f"✅ Đã thêm {len(docs)} docs. Tổng: {len(self.documents)}")

    def clear(self):
        """Xóa toàn bộ index và file hashes."""
        self.documents = []
        self._faiss_index = None
        self._bm25 = None
        self._file_hashes = set()
        for p in [self._docs_path, self._faiss_bin_path, self._hashes_path]:
            if p.exists():
                p.unlink()
        logger.info("✅ Đã xóa toàn bộ index")

    def retrieve(self, query: str, k: Optional[int] = None) -> list[Document]:
        """
        Retrieve top-k docs liên quan nhất.
        Pipeline: Hybrid search (BM25 + FAISS) → RRF fusion → Rerank → Top-k
        """
        if not self.documents:
            logger.warning("Index rỗng, chưa có tài liệu nào được thêm.")
            return []

        k = k or self.config.retrieval_k
        fetch_k = min(self.config.retrieval_fetch_k, len(self.documents))

        if self.config.use_hybrid_search:
            candidates = self._hybrid_retrieve(query, fetch_k)
        else:
            candidates = self._dense_retrieve(query, fetch_k)

        if self.config.use_reranker and len(candidates) > k:
            candidates = self._rerank(query, candidates, k)
        else:
            candidates = candidates[:k]

        return candidates

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    def _dense_retrieve(self, query: str, k: int) -> list[Document]:
        """Pure vector search với FAISS."""
        import numpy as np
        faiss_idx = self._get_faiss()
        if faiss_idx is None:
            return []

        import faiss as faiss_lib
        q_embed = np.array([self._embed_query(query)], dtype="float32")
        faiss_lib.normalize_L2(q_embed)
        scores, indices = faiss_idx.search(q_embed, min(k, len(self.documents)))
        return [self.documents[i] for i in indices[0] if i >= 0]

    def _bm25_retrieve(self, query: str, k: int) -> list[Document]:
        """Keyword search với BM25."""
        bm25 = self._get_bm25()
        if bm25 is None:
            return []
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices]

    def _hybrid_retrieve(self, query: str, k: int) -> list[Document]:
        """
        Kết hợp BM25 + dense search dùng Reciprocal Rank Fusion (RRF).
        RRF score = Σ 1/(rank + 60) cho mỗi result list.
        """
        dense_results = self._dense_retrieve(query, k)
        bm25_results = self._bm25_retrieve(query, k)

        # RRF fusion
        rrf_scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}

        def get_doc_id(doc: Document) -> int:
            return id(doc) if doc not in self.documents else self.documents.index(doc)

        for rank, doc in enumerate(dense_results):
            try:
                idx = self.documents.index(doc)
            except ValueError:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rank + 60)
            doc_map[idx] = doc

        for rank, doc in enumerate(bm25_results):
            try:
                idx = self.documents.index(doc)
            except ValueError:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rank + 60)
            doc_map[idx] = doc

        # Sort theo RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return [doc_map[i] for i in sorted_indices[:k]]

    def _rerank(self, query: str, docs: list[Document], k: int) -> list[Document]:
        """FlashRank cross-encoder reranker."""
        reranker = self._get_reranker()
        if reranker is None:
            return docs[:k]

        try:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
            request = RerankRequest(query=query, passages=passages)
            results = reranker.rerank(request)
            # flashrank trả về list of dict hoặc object tuỳ version
            reranked_docs = []
            for r in results[:k]:
                idx = r["id"] if isinstance(r, dict) else r.id
                reranked_docs.append(docs[idx])
            logger.debug(f"Reranked {len(docs)} → {len(reranked_docs)} docs")
            return reranked_docs
        except Exception as e:
            logger.warning(f"Rerank thất bại: {e}")
            return docs[:k]

    # ------------------------------------------------------------------
    # Persistent storage — R1: FAISS binary + documents.pkl
    # ------------------------------------------------------------------

    def _save_faiss_index(self):
        """Lưu FAISS index xuống disk dạng binary."""
        if self._faiss_index is None:
            return
        try:
            import faiss
            faiss.write_index(self._faiss_index, str(self._faiss_bin_path))
            logger.debug(f"Đã lưu FAISS index → {self._faiss_bin_path}")
        except Exception as e:
            logger.warning(f"Không thể lưu FAISS index: {e}")

    def _load_faiss_index(self) -> bool:
        """
        Load FAISS index từ disk.
        Trả về True nếu thành công và số vector khớp với doc count.
        """
        if not self._faiss_bin_path.exists():
            return False
        try:
            import faiss
            index = faiss.read_index(str(self._faiss_bin_path))
            # Kiểm tra số vector khớp với số docs
            if index.ntotal != len(self.documents):
                logger.warning(
                    f"FAISS index ({index.ntotal} vectors) không khớp với "
                    f"documents ({len(self.documents)}). Sẽ rebuild."
                )
                self._faiss_bin_path.unlink(missing_ok=True)
                return False
            self._faiss_index = index
            logger.info(f"✅ Loaded FAISS index từ disk ({index.ntotal} vectors)")
            return True
        except Exception as e:
            logger.warning(f"FAISS index bị lỗi: {e}. Xóa và rebuild.")
            self._faiss_bin_path.unlink(missing_ok=True)
            return False

    def _save_persistent(self):
        """Lưu documents.pkl và FAISS index xuống disk."""
        try:
            with open(self._docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            logger.debug(f"Đã lưu {len(self.documents)} docs → {self._docs_path}")
        except Exception as e:
            logger.warning(f"Không thể lưu documents: {e}")

    def _save_hashes(self):
        """Lưu file hashes xuống disk."""
        try:
            with open(self._hashes_path, "wb") as f:
                pickle.dump(self._file_hashes, f)
        except Exception as e:
            logger.warning(f"Không thể lưu file hashes: {e}")

    def _load_persistent(self):
        """
        Load documents từ disk, sau đó load FAISS index.
        Nếu FAISS index không hợp lệ → rebuild (chỉ embed lại, không đọc file).
        """
        if not self._docs_path.exists():
            return
        try:
            with open(self._docs_path, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"✅ Loaded {len(self.documents)} docs từ persistent store")
        except Exception as e:
            logger.warning(f"Không thể load documents: {e}. Bắt đầu từ đầu.")
            self.documents = []
            return

        # Load file hashes (R4)
        if self._hashes_path.exists():
            try:
                with open(self._hashes_path, "rb") as f:
                    self._file_hashes = pickle.load(f)
                logger.debug(f"Loaded {len(self._file_hashes)} file hashes")
            except Exception:
                self._file_hashes = set()

        if not self.documents:
            return

        # R1: Thử load FAISS từ disk trước
        if not self._load_faiss_index():
            logger.info("Rebuild FAISS index từ documents...")
            self._build_faiss()

        # BM25 luôn rebuild từ text (rất nhanh, < 1 giây)
        self._build_bm25()

    @property
    def doc_count(self) -> int:
        return len(self.documents)


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    logging.basicConfig(level=logging.INFO)
    from config import config

    ret = HybridRAGRetriever(config)
    print(f"Docs hiện tại: {ret.doc_count}")

    # Test với dummy docs
    from langchain_core.documents import Document
    test_docs = [
        Document(page_content="RAG là kỹ thuật kết hợp retrieval và generation."),
        Document(page_content="Llama 3 là model ngôn ngữ lớn của Meta AI."),
        Document(page_content="FAISS là thư viện tìm kiếm vector của Facebook."),
    ]
    ret.add_documents(test_docs)
    results = ret.retrieve("RAG là gì?", k=2)
    print(f"\nKết quả tìm kiếm:")
    for r in results:
        print(f"  - {r.page_content}")
