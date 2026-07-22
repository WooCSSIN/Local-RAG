"""
decomposer.py — Query Decomposition Agent
Analyzes complex questions and splits them into simpler sub-queries.
Each sub-query is retrieved independently, then results are merged.
"""
import logging
from dataclasses import dataclass

from src.prompts import DECOMPOSE_CHECK_PROMPT, DECOMPOSE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class DecomposeResult:
    """Result of query decomposition analysis."""
    should_decompose: bool
    sub_queries: list[str]
    original_query: str


class QueryDecomposer:
    """
    Analyze questions and decompose complex ones into sub-queries.

    A question needs decomposition when:
    - It compares 2+ concepts
    - It asks about multiple aspects of the same topic
    - It requires information from different parts of documents
    """

    def should_decompose(self, question: str, llm) -> bool:
        """
        Quick check if a question needs decomposition.
        Uses a single LLM call returning YES/NO.
        """
        try:
            prompt = DECOMPOSE_CHECK_PROMPT.format(question=question)
            response = llm.invoke(prompt).content.strip().upper()
            result = "YES" in response
            if result:
                logger.info(f"Query flagged for decomposition: '{question[:60]}...'")
            return result
        except Exception as e:
            logger.warning(f"Decompose check failed: {e}. Skipping decomposition.")
            return False

    def decompose(self, question: str, llm) -> list[str]:
        """
        Decompose a complex question into 2-4 sub-queries.
        Returns list of simpler, focused questions.
        """
        try:
            prompt = DECOMPOSE_PROMPT.format(question=question)
            response = llm.invoke(prompt).content.strip()

            # Parse response — each line is a sub-query
            lines = [line.strip() for line in response.split("\n")]
            sub_queries = [
                line.lstrip("-•0123456789.) ")
                for line in lines
                if line.strip() and len(line.strip()) > 10
            ]

            # Validate: need at least 2 sub-queries, max 4
            if len(sub_queries) < 2:
                logger.info("Decomposition produced < 2 sub-queries, skipping.")
                return [question]

            sub_queries = sub_queries[:4]
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            return sub_queries

        except Exception as e:
            logger.warning(f"Decomposition failed: {e}. Using original query.")
            return [question]

    def analyze(self, question: str, llm) -> DecomposeResult:
        """
        Full analysis: check if decomposition needed, then decompose.
        Returns DecomposeResult with all info.
        """
        if not self.should_decompose(question, llm):
            return DecomposeResult(
                should_decompose=False,
                sub_queries=[question],
                original_query=question,
            )

        sub_queries = self.decompose(question, llm)
        is_decomposed = len(sub_queries) > 1 or sub_queries[0] != question

        return DecomposeResult(
            should_decompose=is_decomposed,
            sub_queries=sub_queries,
            original_query=question,
        )
