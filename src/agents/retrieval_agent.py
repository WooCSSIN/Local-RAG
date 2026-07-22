"""
retrieval_agent.py — Multi-Step Retrieval Agent
Assesses if retrieved context is sufficient and refines queries when needed.
Supports iterative retrieval with query refinement.
"""
import logging
from dataclasses import dataclass

from src.prompts import ASSESS_CONTEXT_PROMPT, REFINE_QUERY_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ContextAssessment:
    """Result of context sufficiency assessment."""
    is_sufficient: bool
    reason: str
    refined_query: str = ""  # Non-empty only when context is insufficient


class RetrievalAgent:
    """
    Assess retrieved context and decide whether more retrieval is needed.

    Workflow:
    1. Assess context sufficiency for the question
    2. If insufficient, refine the query for better retrieval
    3. Return refined query for re-retrieval (max 2 iterations)
    """

    MAX_RETRIEVAL_LOOPS = 2

    def assess_context(
        self, question: str, context_docs: list, llm
    ) -> ContextAssessment:
        """
        Assess whether retrieved context is sufficient to answer the question.

        Args:
            question: The user's question
            context_docs: List of retrieved Document objects
            llm: LLM instance for assessment

        Returns:
            ContextAssessment with sufficiency verdict
        """
        if not context_docs:
            return ContextAssessment(
                is_sufficient=False,
                reason="No documents retrieved",
            )

        # Build context summary for assessment (truncate to avoid token limits)
        context_summary = self._build_context_summary(context_docs)

        try:
            prompt = ASSESS_CONTEXT_PROMPT.format(
                question=question,
                context=context_summary[:2000],
            )
            response = llm.invoke(prompt).content.strip().upper()

            is_sufficient = "SUFFICIENT" in response and "INSUFFICIENT" not in response

            if is_sufficient:
                logger.info("Context assessed as SUFFICIENT")
                return ContextAssessment(
                    is_sufficient=True,
                    reason="Context contains relevant information",
                )
            else:
                logger.info("Context assessed as INSUFFICIENT — will refine query")
                return ContextAssessment(
                    is_sufficient=False,
                    reason="Context lacks sufficient information",
                )

        except Exception as e:
            logger.warning(f"Context assessment failed: {e}. Defaulting to sufficient.")
            return ContextAssessment(
                is_sufficient=True,
                reason=f"Assessment error: {e} — defaulting to sufficient",
            )

    def refine_query(
        self, question: str, context_docs: list, llm
    ) -> str:
        """
        Refine the query for better retrieval when context is insufficient.

        Args:
            question: Original question
            context_docs: Previously retrieved documents
            llm: LLM instance for query refinement

        Returns:
            Refined query string
        """
        context_summary = self._build_context_summary(context_docs)

        try:
            prompt = REFINE_QUERY_PROMPT.format(
                question=question,
                context_summary=context_summary[:1000],
            )
            refined = llm.invoke(prompt).content.strip()

            # Validate refined query
            if refined and len(refined) > 5 and refined != question:
                logger.info(f"Query refined: '{question[:50]}' -> '{refined[:50]}'")
                return refined
            else:
                logger.info("Refined query too similar or invalid, using original")
                return question

        except Exception as e:
            logger.warning(f"Query refinement failed: {e}. Using original query.")
            return question

    def assess_and_refine(
        self, question: str, context_docs: list, llm
    ) -> ContextAssessment:
        """
        Combined assess + refine in one call.
        If insufficient, automatically refines the query.
        """
        assessment = self.assess_context(question, context_docs, llm)

        if not assessment.is_sufficient:
            refined = self.refine_query(question, context_docs, llm)
            assessment.refined_query = refined

        return assessment

    @staticmethod
    def _build_context_summary(docs: list, max_chars: int = 2000) -> str:
        """Build a truncated context summary for assessment prompts."""
        if not docs:
            return "(Không có tài liệu)"

        parts = []
        total_len = 0
        for doc in docs:
            content = doc.page_content[:300]
            filename = doc.metadata.get("filename", "unknown")
            part = f"[{filename}] {content}"
            if total_len + len(part) > max_chars:
                parts.append("... (additional documents truncated)")
                break
            parts.append(part)
            total_len += len(part)

        return "\n\n".join(parts)
