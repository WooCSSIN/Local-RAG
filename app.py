"""
app.py — Local RAG Chatbot v2
Entry point. Chạy: python app.py
UI: Gradio 6.x, streaming, multi-file, PDF preview
Supports: Groq (cloud free) | Ollama (local) | OpenAI
"""
import os
import sys
import logging
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from src.utils import setup_logging, render_pdf_page, format_sources
from src.document_processor import DocumentProcessor
from src.retriever import HybridRAGRetriever
from src.llm_factory import get_llm, get_provider_name, invalidate_llm_cache
from src.session_manager import SessionManager

setup_logging("INFO")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Khởi tạo components
# ------------------------------------------------------------------
processor = DocumentProcessor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
retriever = HybridRAGRetriever(config)
session_mgr = SessionManager()  # R8+R9: session manager


# ------------------------------------------------------------------
# Upload & Index — R3: Progress, R4: Duplicate check
# ------------------------------------------------------------------
def upload_and_index(files, progress=None):
    if not files:
        return "Chưa chọn file.", None

    results = []
    preview_img = None
    file_list = files if isinstance(files, list) else [files]
    total = len(file_list)

    for idx, file in enumerate(file_list):
        file_path = file.name if hasattr(file, "name") else str(file)
        fname = Path(file_path).name

        # R3: Progress update
        status_now = f"⏳ Đang xử lý {idx+1}/{total}: {fname}"
        yield status_now, preview_img

        # R4: Duplicate check
        try:
            if retriever.is_duplicate(file_path):
                results.append(f"⚠️ {fname}: Đã được index trước đó. Bỏ qua.")
                continue
        except Exception:
            pass  # Nếu không hash được, tiếp tục xử lý

        try:
            docs = processor.process(file_path)
            retriever.add_documents(docs)
            retriever.register_file(file_path)  # R4: Lưu hash
            results.append(f"✅ {fname}: {len(docs)} chunks")
            if file_path.lower().endswith(".pdf") and preview_img is None:
                preview_img = render_pdf_page(file_path, 0)
        except Exception as e:
            results.append(f"❌ {fname}: {e}")
            logger.error(f"Upload error {file_path}: {e}")

    # R3: Final status
    status = "\n".join(results)
    status += f"\n\n📊 Tổng docs trong index: {retriever.doc_count}"
    yield status, preview_img


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

        # Build prompt — R6: tiếng Việt có dấu + SystemMessage
        from langchain_core.messages import SystemMessage

        system_msg = SystemMessage(content=(
            "Bạn là trợ lý thông minh chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp. "
            "Luôn trả lời bằng ngôn ngữ của câu hỏi (tiếng Việt hoặc tiếng Anh). "
            "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ thay vì đoán. "
            "Khi trích dẫn thông tin, hãy đề cập nguồn tài liệu."
        ))

        if context_docs:
            context_str = "\n\n---\n\n".join([
                f"[{doc.metadata.get('filename', 'tài liệu')}"
                + (f" trang {doc.metadata['page']}" if doc.metadata.get("page") is not None else "")
                + f"]\n{doc.page_content}"
                for doc in context_docs
            ])
            user_content = (
                f"=== NGỮ CẢNH ===\n{context_str}\n================\n\n"
                + (f"=== LỊCH SỬ ===\n{history_str}\n================\n\n" if history_str else "")
                + f"Câu hỏi: {message}"
            )
        else:
            user_content = (
                (f"Lịch sử:\n{history_str}\n\n" if history_str else "")
                + f"Câu hỏi: {message}"
            )

        messages = [system_msg, HumanMessage(content=user_content)]

        # Stream LLM response — R2: cached LLM, R7: stable streaming
        try:
            llm = get_llm(config)
            full_answer = ""
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer += token
                    yield full_answer
        except Exception as conn_err:
            logger.warning(f"LLM connection error, refresh cache: {conn_err}")
            try:
                llm = get_llm(config, force_refresh=True)
                full_answer = ""
                for chunk in llm.stream(messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield full_answer
            except Exception as e:
                raise e

        # Append nguồn tài liệu
        if source_info:
            full_answer += f"\n\n---\n**Nguồn:**\n{source_info}"
            yield full_answer

        # R8: Lưu cặp hội thoại vào session persistent
        try:
            sid = session_mgr.ensure_current()
            session_mgr.add_message(sid, "user", message)
            session_mgr.add_message(sid, "assistant", full_answer)
        except Exception as e:
            logger.warning(f"Không thể lưu session: {e}")

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        yield f"❌ Lỗi: {str(e)}"


# ------------------------------------------------------------------
# Other handlers
# ------------------------------------------------------------------
def clear_index_fn():
    retriever.clear()
    return "🗑️ Đã xóa index. Upload tài liệu mới để bắt đầu.", None


# ------------------------------------------------------------------
# R5: Export chat ra file
# ------------------------------------------------------------------
def export_chat(history, fmt="markdown"):
    """Xuất lịch sử chat ra file TXT hoặc Markdown."""
    if not history:
        return None, "⚠️ Chưa có nội dung chat để xuất."

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = []

    if fmt == "txt":
        filename = f"chat_export_{now}.txt"
        for turn in history:
            if isinstance(turn, dict):
                role = "User" if turn.get("role") == "user" else "Assistant"
                content = turn.get("content", "")
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                lines.append(f"[{ts}] {role}: {content}")
                lines.append("")
        content_str = "\n".join(lines)
    else:
        filename = f"chat_export_{now}.md"
        lines.append(f"# Chat Export — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        for turn in history:
            if isinstance(turn, dict):
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    lines.append(f"**User:** {content}\n")
                elif role == "assistant":
                    lines.append(f"{content}\n")
                    lines.append("---\n")
        content_str = "\n".join(lines)

    # Lưu ra file tạm
    export_dir = Path("./chat_exports")
    export_dir.mkdir(exist_ok=True)
    export_path = export_dir / filename
    export_path.write_text(content_str, encoding="utf-8")
    return str(export_path), f"✅ Đã xuất chat → {filename}"


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
# Gradio UI — R3/R4/R5 integrated
# ------------------------------------------------------------------
def build_ui():
    import gradio as gr

    provider_label = get_provider_name(config)

    with gr.Blocks(title="RAG Chatbot v3") as demo:

        gr.Markdown(f"""
# 🤖 RAG Chatbot v3
Chat với tài liệu — Hybrid BM25+FAISS+Reranker+LangGraph
**LLM:** `{provider_label}` | **Embed:** `{config.embedding_model.split('/')[-1]}`
        """)

        with gr.Row():
            # ---- Cột trái ----
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📁 Upload tài liệu")
                file_input = gr.File(
                    label="Chọn file (PDF, DOCX, XLSX, TXT, HTML, MD, ảnh)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".xlsx", ".txt", ".html", ".md",
                                ".png", ".jpg", ".jpeg"],
                )
                upload_btn = gr.Button("🔄 Xử lý & Index", variant="primary")
                upload_status = gr.Textbox(label="Trạng thái", interactive=False, lines=5)
                pdf_preview = gr.Image(label="PDF Preview", height=350)

                gr.Markdown("### ⚙️ Hệ thống")
                status_btn = gr.Button("🔍 Kiểm tra")
                system_status = gr.Textbox(label="Status", interactive=False, lines=8)
                clear_btn = gr.Button("🗑️ Xóa index", variant="stop")

            # ---- Cột phải: Chat ----
            with gr.Column(scale=3):
                gr.Markdown("### 💬 Chat")

                # R9: Session management panel
                with gr.Accordion("📂 Quản lý phiên chat", open=False):
                    with gr.Row():
                        new_session_btn = gr.Button("➕ Phiên mới", size="sm", scale=1)
                        session_name_input = gr.Textbox(
                            placeholder="Tên phiên mới...",
                            show_label=False, scale=3, container=False
                        )
                    session_dropdown = gr.Dropdown(
                        choices=[], label="Chọn phiên", interactive=True
                    )
                    with gr.Row():
                        rename_input = gr.Textbox(
                            placeholder="Tên mới...", show_label=False, scale=3, container=False
                        )
                        rename_btn = gr.Button("✏️ Đổi tên", size="sm", scale=1)
                        delete_session_btn = gr.Button("🗑️ Xóa phiên", size="sm", scale=1, variant="stop")

                chatbot_ui = gr.Chatbot(height=430)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Nhập câu hỏi, nhấn Enter...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Gửi ▶", variant="primary", scale=1)

                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Xóa chat", size="sm", scale=1)
                    # R5: Export chat
                    export_fmt = gr.Radio(
                        choices=["markdown", "txt"],
                        value="markdown",
                        label="Định dạng",
                        scale=1,
                    )
                    export_btn = gr.Button("📥 Xuất chat", size="sm", scale=1)
                    export_file = gr.File(label="File tải về", scale=2, visible=False)
                    export_status = gr.Textbox(show_label=False, scale=2, interactive=False, visible=False)

                gr.Examples(
                    examples=[
                        "Tóm tắt nội dung chính của tài liệu.",
                        "Các phương pháp RAG nào được đề cập?",
                        "Giải thích về retrieval augmented generation.",
                        "Kết luận của tài liệu là gì?",
                    ],
                    inputs=msg_box,
                    label="Câu hỏi mẫu",
                )

        # ---- Events ----

        # R3+R4: Upload với streaming progress
        upload_btn.click(
            fn=upload_and_index,
            inputs=[file_input],
            outputs=[upload_status, pdf_preview],
        )

        # R8+R9: Session management events
        def _refresh_sessions():
            sessions = session_mgr.list_sessions()
            choices = [f"{s['name']} ({s['msg_count']} msgs)" for s in sessions]
            ids = [s["id"] for s in sessions]
            # Map label -> id
            choices_with_id = [(s["name"] + f" [{s['id']}]", s["id"]) for s in sessions]
            return gr.update(choices=choices_with_id, value=None)

        def _new_session(name):
            sid = session_mgr.create_session(name.strip() if name and name.strip() else None)
            return [], _refresh_sessions()

        def _switch_session(sid_label):
            if not sid_label:
                return []
            # sid_label dạng "Tên [sid]"
            import re
            match = re.search(r"\[([a-f0-9]+)\]$", str(sid_label))
            if match:
                sid = match.group(1)
                return session_mgr.switch_to(sid)
            return []

        def _rename_session(sid_label, new_name):
            if not sid_label or not new_name.strip():
                return _refresh_sessions()
            import re
            match = re.search(r"\[([a-f0-9]+)\]$", str(sid_label))
            if match:
                session_mgr.rename_session(match.group(1), new_name.strip())
            return _refresh_sessions()

        def _delete_session(sid_label):
            if not sid_label:
                return [], _refresh_sessions()
            import re
            match = re.search(r"\[([a-f0-9]+)\]$", str(sid_label))
            if match:
                sid = match.group(1)
                if sid == session_mgr.current_id:
                    session_mgr.create_session()
                session_mgr.delete_session(sid)
            return [], _refresh_sessions()

        new_session_btn.click(fn=_new_session, inputs=[session_name_input], outputs=[chatbot_ui, session_dropdown])
        session_dropdown.change(fn=_switch_session, inputs=[session_dropdown], outputs=[chatbot_ui])
        rename_btn.click(fn=_rename_session, inputs=[session_dropdown, rename_input], outputs=[session_dropdown])
        delete_session_btn.click(fn=_delete_session, inputs=[session_dropdown], outputs=[chatbot_ui, session_dropdown])

        # Load sessions on startup
        demo.load(fn=lambda: _refresh_sessions(), outputs=[session_dropdown])

        # Chat
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
            last_user_msg = None
            for turn in reversed(history):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    last_user_msg = turn.get("content", "")
                    break
            if not last_user_msg:
                yield history
                return
            prev_history = history[:-2] if len(history) >= 2 else []
            history = list(history)
            # R7: ổn định streaming — catch exception mid-stream
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

        # R5: Export chat
        def _export(history, fmt):
            path, msg = export_chat(history, fmt)
            if path:
                return (
                    gr.update(value=path, visible=True),
                    gr.update(value=msg, visible=True),
                )
            return (
                gr.update(visible=False),
                gr.update(value=msg, visible=True),
            )

        export_btn.click(
            fn=_export,
            inputs=[chatbot_ui, export_fmt],
            outputs=[export_file, export_status],
        )

        status_btn.click(fn=system_status_fn, outputs=[system_status])
        clear_btn.click(fn=clear_index_fn, outputs=[upload_status, pdf_preview])

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
