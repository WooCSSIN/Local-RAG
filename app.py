"""
app.py — Local RAG Chatbot v2
Entry point. Chạy: python app.py
UI: Gradio 6.x, streaming, multi-file, PDF preview
Supports: Groq (cloud free) | Ollama (local) | OpenAI
"""
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from src.utils import setup_logging, render_pdf_page, format_sources
from src.document_processor import DocumentProcessor
from src.retriever import HybridRAGRetriever
from src.llm_factory import get_llm, get_provider_name

setup_logging("INFO")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Khởi tạo components
# ------------------------------------------------------------------
processor = DocumentProcessor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
retriever = HybridRAGRetriever(config)


# ------------------------------------------------------------------
# Upload & Index
# ------------------------------------------------------------------
def upload_and_index(files):
    if not files:
        return "Chưa chọn file.", None

    results = []
    preview_img = None
    file_list = files if isinstance(files, list) else [files]

    for file in file_list:
        file_path = file.name if hasattr(file, "name") else str(file)
        try:
            docs = processor.process(file_path)
            retriever.add_documents(docs)
            results.append(f"OK {Path(file_path).name}: {len(docs)} chunks")
            if file_path.lower().endswith(".pdf") and preview_img is None:
                preview_img = render_pdf_page(file_path, 0)
        except Exception as e:
            results.append(f"LOI {Path(file_path).name}: {e}")
            logger.error(f"Upload error {file_path}: {e}")

    status = "\n".join(results)
    status += f"\n\nTong docs trong index: {retriever.doc_count}"
    return status, preview_img


# ------------------------------------------------------------------
# Chat function — streaming
# ------------------------------------------------------------------
def chat_fn(message: str, history: list):
    """Generator — yield partial response cho Gradio streaming."""
    message = str(message) if message else ""
    if not message.strip():
        return

    from langchain_core.messages import HumanMessage, AIMessage

    # Convert Gradio 6 history (list of dicts) -> LangChain messages
    lc_history = []
    for turn in (history or []):
        if isinstance(turn, dict):
            role = turn.get("role", "")
            content = turn.get("content") or ""
            if role == "user" and content:
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                lc_history.append(AIMessage(content=content))
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            # fallback cho format cũ
            human, ai = turn
            if human:
                lc_history.append(HumanMessage(content=str(human)))
            if ai:
                lc_history.append(AIMessage(content=str(ai)))

    try:
        # Retrieve docs
        context_docs = []
        source_info = ""
        if retriever.doc_count > 0:
            try:
                context_docs = retriever.retrieve(message)
                if context_docs:
                    source_info = format_sources(context_docs)
            except Exception as e:
                logger.warning(f"Retrieve error: {e}")

        # Build history string
        history_str = ""
        if lc_history:
            recent = lc_history[-config.max_chat_history:]
            history_str = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in recent
            ])

        # Build prompt
        if context_docs:
            context_str = "\n\n---\n\n".join([
                "[" + doc.metadata.get("filename", "doc")
                + (f" trang {doc.metadata['page']}" if doc.metadata.get("page") is not None else "")
                + "]\n" + doc.page_content
                for doc in context_docs
            ])
            prompt = (
                "Ban la tro ly thong minh. Tra loi dua tren ngu canh.\n"
                "Neu khong du thong tin hay noi ro. Tra loi bang ngon ngu cau hoi.\n\n"
                f"=== NGU CANH ===\n{context_str}\n================\n\n"
                + (f"=== LICH SU ===\n{history_str}\n================\n\n" if history_str else "")
                + f"Cau hoi: {message}\n\nTra loi:"
            )
        else:
            prompt = (
                (f"Lich su:\n{history_str}\n\n" if history_str else "")
                + f"Cau hoi: {message}\nTra loi:"
            )

        # Stream LLM response
        llm = get_llm(config)
        full_answer = ""
        for chunk in llm.stream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += token
            yield full_answer

        # Append sources
        if source_info:
            full_answer += f"\n\n---\nNguon:\n{source_info}"
            yield full_answer

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        yield f"Loi: {str(e)}"


# ------------------------------------------------------------------
# Other handlers
# ------------------------------------------------------------------
def clear_index_fn():
    retriever.clear()
    return "Da xoa index. Upload tai lieu moi de bat dau.", None


def system_status_fn():
    lines = []
    provider = get_provider_name(config)
    lines.append(f"LLM Provider : {provider}")
    lines.append(f"Docs in index: {retriever.doc_count}")
    lines.append(f"Embedding    : {config.embedding_model.split('/')[-1]}")
    lines.append(f"Hybrid search: {config.use_hybrid_search}")
    lines.append(f"Reranker     : {config.use_reranker}")
    lines.append(f"Chunk size   : {config.chunk_size}")

    # Test LLM connection
    try:
        llm = get_llm(config)
        lines.append("LLM status   : OK (config valid)")
    except Exception as e:
        lines.append(f"LLM status   : ERROR - {e}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
def build_ui():
    import gradio as gr

    provider_label = get_provider_name(config)

    with gr.Blocks(title="Local RAG Chatbot v2") as demo:

        gr.Markdown(f"""
# RAG Chatbot v2
Chat voi tai lieu — Hybrid BM25+FAISS+Reranker+LangGraph
**LLM:** `{provider_label}` | **Embed:** `{config.embedding_model.split('/')[-1]}`
        """)

        with gr.Row():
            # ---- Left column ----
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Upload tai lieu")
                file_input = gr.File(
                    label="Chon file (PDF, TXT, HTML)",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".html", ".md"],
                )
                upload_btn = gr.Button("Xu ly & Index", variant="primary")
                upload_status = gr.Textbox(label="Trang thai", interactive=False, lines=4)
                pdf_preview = gr.Image(label="PDF Preview", height=350)

                gr.Markdown("### He thong")
                status_btn = gr.Button("Kiem tra")
                system_status = gr.Textbox(label="Status", interactive=False, lines=8)
                clear_btn = gr.Button("Xoa index", variant="stop")

            # ---- Right column: Chat ----
            with gr.Column(scale=3):
                gr.Markdown("### Chat")
                chatbot_ui = gr.Chatbot(height=500)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Nhap cau hoi, nhan Enter...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Gui", variant="primary", scale=1)
                with gr.Row():
                    clear_chat_btn = gr.Button("Xoa chat", size="sm")

                gr.Examples(
                    examples=[
                        "Tom tat noi dung chinh cua tai lieu.",
                        "Cac phuong phap RAG nao duoc de cap?",
                        "Giai thich ve retrieval augmented generation.",
                        "Ket luan cua tai lieu la gi?",
                    ],
                    inputs=msg_box,
                )

        # ---- Events ----
        upload_btn.click(fn=upload_and_index, inputs=[file_input], outputs=[upload_status, pdf_preview])

        def _submit(message, history):
            if not message or not str(message).strip():
                return history, ""
            history = list(history or [])
            history.append({"role": "user", "content": str(message)})
            history.append({"role": "assistant", "content": "⏳ Đang xử lý..."})
            return history, ""

        def _stream(history):
            if not history or len(history) < 2:
                yield history
                return
            # Tìm user message cuối cùng (kế trước assistant)
            last_user_msg = None
            for turn in reversed(history):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    last_user_msg = turn.get("content", "")
                    break
            if not last_user_msg:
                yield history
                return

            # Lấy history trước đó (không gồm cặp user+assistant placeholder cuối)
            prev_history = history[:-2] if len(history) >= 2 else []

            history = list(history)
            try:
                for partial in chat_fn(last_user_msg, prev_history):
                    history[-1] = {"role": "assistant", "content": partial}
                    yield history
            except Exception as e:
                history[-1] = {"role": "assistant", "content": f"❌ Lỗi: {e}"}
                yield history

        msg_box.submit(
            fn=_submit, inputs=[msg_box, chatbot_ui], outputs=[chatbot_ui, msg_box]
        ).then(fn=_stream, inputs=[chatbot_ui], outputs=[chatbot_ui])

        send_btn.click(
            fn=_submit, inputs=[msg_box, chatbot_ui], outputs=[chatbot_ui, msg_box]
        ).then(fn=_stream, inputs=[chatbot_ui], outputs=[chatbot_ui])

        clear_chat_btn.click(lambda: [], outputs=[chatbot_ui])
        status_btn.click(fn=system_status_fn, outputs=[system_status])
        clear_btn.click(fn=clear_index_fn, outputs=[upload_status, pdf_preview])
        demo.load(fn=system_status_fn, outputs=[system_status])

    return demo


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import gradio as gr

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=config.server_host)
    parser.add_argument("--port", type=int, default=config.server_port)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Local RAG Chatbot v2")
    logger.info(f"Provider : {get_provider_name(config)}")
    logger.info(f"Embedding: {config.embedding_model}")
    logger.info(f"Docs     : {retriever.doc_count} in index")
    logger.info("=" * 50)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )
