"""
agentic_graph.py — Full Agentic LangGraph Orchestration
Integrates all agents: decomposer, grader, retrieval_agent, tool_agent.

Graph structure:
  route_question
    → [YES] decompose_check
      → [simple] rewrite → retrieve → assess_context → [sufficient] generate → grade_answer → END
      → [complex] decompose → parallel_retrieve → merge_context → generate → grade_answer → END
    → [NO] generate (chatty response) → END

  assess_context → [insufficient] refine_query → retrieve (max 2 loops)
  grade_answer → [fail] rewrite → retrieve (max 2 loops)
  tool_use → when local retrieval insufficient, call web search
"""
import logging
import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Agentic Graph State
# ------------------------------------------------------------------

class AgenticState(TypedDict):
    """Extended state for the agentic RAG graph."""
    question: str
    original_question: str
    chat_history: Annotated[list[BaseMessage], operator.add]
    context: list  # retrieved Documents
    answer: str
    source_pages: list
    needs_retrieval: bool
    needs_decomposition: bool
    sub_queries: list[str]
    retrieval_loop_count: int
    grade_loop_count: int
    context_sufficient: bool
    use_web_tool: bool
    web_content: str
    grade_passed: bool


# ------------------------------------------------------------------
# Build Agentic Graph
# ------------------------------------------------------------------

def build_agentic_graph(retriever, config):
    """
    Create the full agentic RAG graph with all agents.

    Args:
        retriever: HybridRAGRetriever instance
        config: RAGConfig instance

    Returns:
        Compiled LangGraph
    """
    try:
        from langgraph.graph import StateGraph, END
        from src.llm_factory import get_llm
        from src.agents.decomposer import QueryDecomposer
        from src.agents.grader import AnswerGrader
        from src.agents.retrieval_agent import RetrievalAgent
        from src.agents.tools import ToolAgent
        from src.prompts import (
            ROUTE_PROMPT, REWRITE_PROMPT,
            GENERATE_WITH_CONTEXT_PROMPT, GENERATE_NO_CONTEXT_PROMPT,
            build_context_string, build_history_string,
        )
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    llm = get_llm(config)
    decomposer = QueryDecomposer()
    grader = AnswerGrader()
    retrieval_agent = RetrievalAgent()
    tool_agent = ToolAgent()

    # ------------------------------------------------------------------
    # Node 1: Route — decide if retrieval is needed
    # ------------------------------------------------------------------
    def route_question(state: AgenticState) -> AgenticState:
        if retriever.doc_count == 0:
            return {**state, "needs_retrieval": False}

        history_section = build_history_string(state.get("chat_history"), 4)
        prompt = ROUTE_PROMPT.format(
            history_section=f"Lịch sử:\n{history_section}\n\n" if history_section else "",
            question=state["question"],
        )

        try:
            response = llm.invoke(prompt).content.strip().upper()
            needs = "YES" in response
        except Exception as e:
            logger.warning(f"Route node error: {e}. Defaulting to retrieval=True")
            needs = True

        return {**state, "needs_retrieval": needs}

    # ------------------------------------------------------------------
    # Node 2: Decompose check — decide if query needs splitting
    # ------------------------------------------------------------------
    def decompose_check(state: AgenticState) -> AgenticState:
        try:
            result = decomposer.analyze(state["question"], llm)
            return {
                **state,
                "needs_decomposition": result.should_decompose,
                "sub_queries": result.sub_queries,
            }
        except Exception as e:
            logger.warning(f"Decompose check error: {e}")
            return {**state, "needs_decomposition": False, "sub_queries": [state["question"]]}

    # ------------------------------------------------------------------
    # Node 3: Rewrite — make follow-up questions self-contained
    # ------------------------------------------------------------------
    def rewrite_query(state: AgenticState) -> AgenticState:
        history = state.get("chat_history", [])
        if not history:
            return state

        recent = history[-4:]
        history_str = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent
        ])

        try:
            prompt = REWRITE_PROMPT.format(
                history=history_str,
                question=state["question"],
            )
            rewritten = llm.invoke(prompt).content.strip()
            if rewritten and len(rewritten) > 5:
                logger.info(f"Query rewritten: '{state['question'][:40]}' -> '{rewritten[:40]}'")
                return {**state, "question": rewritten}
        except Exception as e:
            logger.warning(f"Rewrite error: {e}")

        return state

    # ------------------------------------------------------------------
    # Node 4: Retrieve — fetch relevant documents
    # ------------------------------------------------------------------
    def retrieve(state: AgenticState) -> AgenticState:
        query = state["question"]
        try:
            docs = retriever.retrieve(query)
            pages = []
            for doc in docs:
                page = doc.metadata.get("page")
                if page is not None and page not in pages:
                    pages.append(page)

            # Merge with existing context (for multi-step retrieval)
            existing = state.get("context", [])
            if existing:
                # Deduplicate by content
                seen_content = {d.page_content for d in existing}
                for doc in docs:
                    if doc.page_content not in seen_content:
                        existing.append(doc)
                        seen_content.add(doc.page_content)
                docs = existing

            loop_count = state.get("retrieval_loop_count", 0) + 1
            return {
                **state,
                "context": docs,
                "source_pages": pages,
                "retrieval_loop_count": loop_count,
            }
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            return {**state, "context": [], "source_pages": []}

    # ------------------------------------------------------------------
    # Node 4b: Parallel retrieve for decomposed queries
    # ------------------------------------------------------------------
    def parallel_retrieve(state: AgenticState) -> AgenticState:
        sub_queries = state.get("sub_queries", [state["question"]])
        all_docs = []
        all_pages = []
        seen_content = set()

        for sq in sub_queries:
            try:
                docs = retriever.retrieve(sq)
                for doc in docs:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)
                        page = doc.metadata.get("page")
                        if page is not None and page not in all_pages:
                            all_pages.append(page)
            except Exception as e:
                logger.warning(f"Parallel retrieve error for '{sq[:30]}': {e}")

        return {
            **state,
            "context": all_docs,
            "source_pages": all_pages,
            "retrieval_loop_count": 1,
        }

    # ------------------------------------------------------------------
    # Node 5: Assess context — check if we have enough info
    # ------------------------------------------------------------------
    def assess_context(state: AgenticState) -> AgenticState:
        loop_count = state.get("retrieval_loop_count", 0)

        # Skip assessment if we've already looped too many times
        if loop_count >= retrieval_agent.MAX_RETRIEVAL_LOOPS:
            return {**state, "context_sufficient": True}

        assessment = retrieval_agent.assess_and_refine(
            state["question"], state.get("context", []), llm
        )

        if not assessment.is_sufficient and assessment.refined_query:
            return {
                **state,
                "context_sufficient": False,
                "question": assessment.refined_query,
            }
        return {**state, "context_sufficient": True}

    # ------------------------------------------------------------------
    # Node 6: Tool use — web search when local docs insufficient
    # ------------------------------------------------------------------
    def tool_use(state: AgenticState) -> AgenticState:
        context_summary = retrieval_agent._build_context_summary(state.get("context", []))
        tool_name = tool_agent.should_use_tool(state["question"], context_summary, llm)

        if tool_name == "web_search":
            result = tool_agent.web_search(state.get("original_question", state["question"]))
            if result.success and result.content:
                return {
                    **state,
                    "use_web_tool": True,
                    "web_content": result.content,
                }

        return {**state, "use_web_tool": False, "web_content": ""}

    # ------------------------------------------------------------------
    # Node 7: Generate — produce answer from context + history
    # ------------------------------------------------------------------
    def generate(state: AgenticState) -> AgenticState:
        history_str = build_history_string(state.get("chat_history"), config.max_chat_history)
        history_section = f"=== LỊCH SỬ ===\n{history_str}\n================\n\n" if history_str else ""

        context_docs = state.get("context", [])
        web_content = state.get("web_content", "")

        if context_docs or web_content:
            context_str = build_context_string(context_docs)
            if web_content:
                context_str += f"\n\n---\n\n[Web Search Results]\n{web_content}"

            prompt = GENERATE_WITH_CONTEXT_PROMPT.format(
                context=context_str,
                history_section=history_section,
                question=state.get("original_question", state["question"]),
            )
        else:
            prompt = GENERATE_NO_CONTEXT_PROMPT.format(
                history_section=history_section,
                question=state.get("original_question", state["question"]),
            )

        try:
            answer = llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.error(f"Generate error: {e}")
            answer = f"Xin lỗi, đã xảy ra lỗi: {str(e)}"

        new_messages = [
            HumanMessage(content=state.get("original_question", state["question"])),
            AIMessage(content=answer),
        ]

        return {**state, "answer": answer, "chat_history": new_messages}

    # ------------------------------------------------------------------
    # Node 8: Grade answer — self-reflection
    # ------------------------------------------------------------------
    def grade_answer(state: AgenticState) -> AgenticState:
        loop_count = state.get("grade_loop_count", 0)

        # Skip grading if already retried max times
        if loop_count >= grader.MAX_RETRIES:
            return {**state, "grade_passed": True}

        context_docs = state.get("context", [])
        context_str = build_context_string(context_docs) if context_docs else ""

        result = grader.grade(
            question=state.get("original_question", state["question"]),
            answer=state.get("answer", ""),
            context=context_str,
            llm=llm,
        )

        return {
            **state,
            "grade_passed": result.overall,
            "grade_loop_count": loop_count + 1,
        }

    # ------------------------------------------------------------------
    # Conditional routing functions
    # ------------------------------------------------------------------
    def route_after_decompose(state: AgenticState) -> str:
        """Route based on whether decomposition is needed."""
        if state.get("needs_decomposition", False):
            return "parallel_retrieve"
        return "rewrite"

    def route_after_assess(state: AgenticState) -> str:
        """Route based on context sufficiency."""
        if state.get("context_sufficient", True):
            return "tool_use"
        return "retrieve"  # Loop back with refined query

    def route_after_grade(state: AgenticState) -> str:
        """Route based on grade result."""
        if state.get("grade_passed", True):
            return "end"
        # Fail — loop back to rewrite + retrieve
        loop_count = state.get("grade_loop_count", 0)
        if loop_count >= grader.MAX_RETRIES:
            return "end"
        return "rewrite"

    def route_after_tool(state: AgenticState) -> str:
        """Always go to generate after tool use."""
        return "generate"

    # ------------------------------------------------------------------
    # Build & compile graph
    # ------------------------------------------------------------------
    workflow = StateGraph(AgenticState)

    # Add all nodes
    workflow.add_node("route", route_question)
    workflow.add_node("decompose_check", decompose_check)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("parallel_retrieve", parallel_retrieve)
    workflow.add_node("assess_context", assess_context)
    workflow.add_node("tool_use", tool_use)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_answer", grade_answer)

    # Entry point
    workflow.set_entry_point("route")

    # Route → decompose_check (if retrieval needed) or generate (if chatty)
    workflow.add_conditional_edges(
        "route",
        lambda s: "decompose_check" if s.get("needs_retrieval", True) else "generate",
        {"decompose_check": "decompose_check", "generate": "generate"},
    )

    # Decompose check → parallel_retrieve (complex) or rewrite (simple)
    workflow.add_conditional_edges(
        "decompose_check",
        route_after_decompose,
        {"parallel_retrieve": "parallel_retrieve", "rewrite": "rewrite"},
    )

    # Rewrite → retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Retrieve → assess_context
    workflow.add_edge("retrieve", "assess_context")

    # Assess context → tool_use (sufficient) or retrieve (insufficient, loop)
    workflow.add_conditional_edges(
        "assess_context",
        route_after_assess,
        {"tool_use": "tool_use", "retrieve": "retrieve"},
    )

    # Parallel retrieve → assess_context (same as single retrieve)
    workflow.add_edge("parallel_retrieve", "assess_context")

    # Tool use → generate
    workflow.add_conditional_edges(
        "tool_use",
        route_after_tool,
        {"generate": "generate"},
    )

    # Generate → grade_answer
    workflow.add_edge("generate", "grade_answer")

    # Grade answer → END (pass) or rewrite (fail, loop)
    workflow.add_conditional_edges(
        "grade_answer",
        route_after_grade,
        {"end": END, "rewrite": "rewrite"},
    )

    return workflow.compile()


# ------------------------------------------------------------------
# Helper — initial state
# ------------------------------------------------------------------

def make_agentic_state(
    question: str,
    chat_history: list[BaseMessage] = None,
) -> AgenticState:
    """Create initial state for agentic graph invocation."""
    return {
        "question": question,
        "original_question": question,
        "chat_history": chat_history or [],
        "context": [],
        "answer": "",
        "source_pages": [],
        "needs_retrieval": True,
        "needs_decomposition": False,
        "sub_queries": [],
        "retrieval_loop_count": 0,
        "grade_loop_count": 0,
        "context_sufficient": False,
        "use_web_tool": False,
        "web_content": "",
        "grade_passed": False,
    }


# ------------------------------------------------------------------
# Streaming-friendly pipeline for app.py
# ------------------------------------------------------------------

def run_agentic_pipeline(
    graph,
    question: str,
    chat_history: list[BaseMessage] = None,
) -> dict:
    """
    Run the agentic graph and return prepared context for streaming generation.

    This runs all orchestration nodes (route, decompose, retrieve, assess, tool)
    but returns the context BEFORE generation, so app.py can stream the final
    LLM response token-by-token.

    Returns dict with keys:
        - needs_retrieval: bool
        - context_docs: list[Document]
        - context_str: str (formatted context for prompt)
        - history_str: str (formatted history for prompt)
        - source_pages: list[int]
        - web_content: str (if web search was used)
        - rewritten_question: str (if query was rewritten)
        - grade_result: dict (if grading was done on final answer)
    """
    from src.prompts import build_context_string, build_history_string

    state = make_agentic_state(question, chat_history)

    # Run the full graph — generate node produces answer
    result = graph.invoke(state)

    context_docs = result.get("context", [])
    web_content = result.get("web_content", "")
    source_pages = result.get("source_pages", [])

    context_str = build_context_string(context_docs) if context_docs else ""
    if web_content:
        context_str += f"\n\n---\n\n[Web Search Results]\n{web_content}"

    history_str = build_history_string(
        result.get("chat_history", [])[:-2],  # exclude the just-added pair
        max_turns=6,
    )

    return {
        "needs_retrieval": result.get("needs_retrieval", True),
        "context_docs": context_docs,
        "context_str": context_str,
        "history_str": history_str,
        "source_pages": source_pages,
        "web_content": web_content,
        "rewritten_question": result.get("question", question),
        "answer": result.get("answer", ""),
        "grade_passed": result.get("grade_passed", None),
        "retrieval_loops": result.get("retrieval_loop_count", 0),
    }


# ------------------------------------------------------------------
# Quick test
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
        print("Start Ollama first: ollama serve")
        sys.exit(1)

    ret = HybridRAGRetriever(config)
    print(f"Docs in index: {ret.doc_count}")

    graph = build_agentic_graph(ret, config)
    state = make_agentic_state("What is RAG and how does it compare to fine-tuning?")
    result = graph.invoke(state)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSource pages: {result['source_pages']}")
    print(f"Grade passed: {result.get('grade_passed')}")
    print(f"Retrieval loops: {result.get('retrieval_loop_count')}")
