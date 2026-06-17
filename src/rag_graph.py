"""
rag_graph.py — LangGraph orchestration
Thay thế ConversationalRetrievalChain cũ.
Graph có 4 nodes: route → rewrite → retrieve → generate
Compatible: LangChain 1.x, LangGraph 1.x
"""
import logging
import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# State schema
# ------------------------------------------------------------------

class GraphState(TypedDict):
    question: str
    chat_history: Annotated[list[BaseMessage], operator.add]
    context: list           # retrieved Documents
    answer: str
    source_pages: list      # trang PDF liên quan để hiển thị
    needs_retrieval: bool


# ------------------------------------------------------------------
# Build graph
# ------------------------------------------------------------------

def build_rag_graph(retriever, config):
    """
    Tạo LangGraph RAG pipeline.
    Args:
        retriever: HybridRAGRetriever instance
        config: RAGConfig instance
    Returns:
        CompiledGraph
    """
    try:
        from langgraph.graph import StateGraph, END
        from src.llm_factory import get_llm
    except ImportError as e:
        raise ImportError(f"Thiếu dependency: {e}")

    llm = get_llm(config)

    # ------------------------------------------------------------------
    # Node 1: Route — quyết định có cần retrieval không
    # ------------------------------------------------------------------
    def route_question(state: GraphState) -> GraphState:
        if retriever.doc_count == 0:
            return {**state, "needs_retrieval": False}

        history_context = ""
        if state.get("chat_history"):
            recent = state["chat_history"][-4:]
            history_context = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in recent
            ])

        prompt = (
            "Xác định xem câu hỏi dưới đây có cần tra cứu tài liệu không.\n"
            "Trả lời chỉ với YES hoặc NO.\n\n"
            "- YES: hỏi về thông tin cụ thể, nội dung tài liệu, dữ liệu\n"
            "- NO: chào hỏi, cảm ơn, câu hỏi chung không liên quan tài liệu\n\n"
            + (f"Lịch sử:\n{history_context}\n\n" if history_context else "")
            + f"Câu hỏi: {state['question']}\nCần retrieval? (YES/NO):"
        )

        try:
            response = llm.invoke(prompt).content.strip().upper()
            needs = "YES" in response
        except Exception as e:
            logger.warning(f"Route node lỗi: {e}. Mặc định retrieval=True")
            needs = True

        return {**state, "needs_retrieval": needs}

    # ------------------------------------------------------------------
    # Node 2: Rewrite — viết lại câu hỏi dựa trên lịch sử
    # ------------------------------------------------------------------
    def rewrite_query(state: GraphState) -> GraphState:
        history = state.get("chat_history", [])
        if not history:
            return state

        recent = history[-4:]
        history_str = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent
        ])

        prompt = (
            "Dựa vào lịch sử hội thoại, viết lại câu hỏi thành câu độc lập, đầy đủ nghĩa.\n"
            "CHỈ trả về câu hỏi đã viết lại, không giải thích thêm.\n\n"
            f"Lịch sử:\n{history_str}\n\n"
            f"Câu hỏi gốc: {state['question']}\n"
            "Câu hỏi viết lại:"
        )

        try:
            rewritten = llm.invoke(prompt).content.strip()
            if rewritten and len(rewritten) > 5:
                logger.debug(f"Query rewritten: '{state['question']}' -> '{rewritten}'")
                return {**state, "question": rewritten}
        except Exception as e:
            logger.warning(f"Rewrite node lỗi: {e}")

        return state

    # ------------------------------------------------------------------
    # Node 3: Retrieve — tìm tài liệu liên quan
    # ------------------------------------------------------------------
    def retrieve(state: GraphState) -> GraphState:
        try:
            docs = retriever.retrieve(state["question"])
            pages = []
            for doc in docs:
                page = doc.metadata.get("page")
                if page is not None and page not in pages:
                    pages.append(page)
            return {**state, "context": docs, "source_pages": pages}
        except Exception as e:
            logger.error(f"Retrieve node lỗi: {e}")
            return {**state, "context": [], "source_pages": []}

    # ------------------------------------------------------------------
    # Node 4: Generate — tạo câu trả lời
    # ------------------------------------------------------------------
    def generate(state: GraphState) -> GraphState:
        history = state.get("chat_history", [])
        history_str = ""
        if history:
            recent = history[-config.max_chat_history:]
            history_str = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in recent
            ])

        context_docs = state.get("context", [])

        if context_docs:
            context_str = "\n\n---\n\n".join([
                f"[Nguồn: {doc.metadata.get('filename', 'unknown')}"
                + (f", trang {doc.metadata.get('page', '')}" if doc.metadata.get("page") is not None else "")
                + f"]\n{doc.page_content}"
                for doc in context_docs
            ])
            prompt = (
                "Bạn là trợ lý thông minh. Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.\n"
                "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ.\n"
                "Trả lời bằng ngôn ngữ của câu hỏi.\n\n"
                f"=== NGỮ CẢNH ===\n{context_str}\n================\n\n"
                + (f"=== LỊCH SỬ ===\n{history_str}\n================\n\n" if history_str else "")
                + f"Câu hỏi: {state['question']}\n\nTrả lời:"
            )
        else:
            prompt = (
                (f"Lịch sử: {history_str}\n" if history_str else "")
                + f"Câu hỏi: {state['question']}\nTrả lời:"
            )

        try:
            answer = llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.error(f"Generate node lỗi: {e}")
            answer = f"Xin lỗi, đã xảy ra lỗi: {str(e)}"

        new_messages = [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ]

        return {**state, "answer": answer, "chat_history": new_messages}

    # ------------------------------------------------------------------
    # Build & compile graph
    # ------------------------------------------------------------------
    workflow = StateGraph(GraphState)

    workflow.add_node("route", route_question)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("route")
    workflow.add_conditional_edges(
        "route",
        lambda s: "rewrite" if s.get("needs_retrieval", True) else "generate",
        {"rewrite": "rewrite", "generate": "generate"}
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ------------------------------------------------------------------
# Helper — initial state
# ------------------------------------------------------------------

def make_initial_state(question: str, chat_history: list[BaseMessage] = None) -> GraphState:
    """Tạo state ban đầu để invoke graph."""
    return {
        "question": question,
        "chat_history": chat_history or [],
        "context": [],
        "answer": "",
        "source_pages": [],
        "needs_retrieval": True,
    }


# ------------------------------------------------------------------
# Quick test (chạy nếu Ollama đang running)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    from config import config
    from src.retriever import HybridRAGRetriever
    from src.utils import check_ollama_connection

    ok, msg = check_ollama_connection(config.llm_base_url)
    print(msg)
    if not ok:
        print("Khởi động Ollama trước: ollama serve")
        sys.exit(1)

    ret = HybridRAGRetriever(config)
    print(f"Docs in index: {ret.doc_count}")

    graph = build_rag_graph(ret, config)
    state = make_initial_state("What is RAG and how does it work?")
    result = graph.invoke(state)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSource pages: {result['source_pages']}")
