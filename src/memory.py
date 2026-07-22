"""
memory.py — Conversation Memory Management
Summarizes old conversation turns to maintain long-term context
without overwhelming the LLM context window.

Strategy:
- Recent turns (last N): kept verbatim
- Older turns: summarized into a short paragraph
- Summary is cached and refreshed only when new turns accumulate
"""
import logging
from typing import Optional

from src.prompts import SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manage long-term conversation context.

    Instead of passing full history (which grows unbounded),
    summarizes older turns and keeps only recent turns verbatim.

    Usage:
        memory = ConversationMemory(max_recent_turns=4)
        context = memory.get_context(history, llm)
        # context contains: summary of old turns + recent turns verbatim
    """

    def __init__(self, max_recent_turns: int = 4):
        self.max_recent_turns = max_recent_turns
        self._cached_summary: Optional[str] = None
        self._summary_turn_count: int = 0  # how many turns the summary covers

    def get_context(
        self,
        history: list,
        llm=None,
    ) -> str:
        """
        Build conversation context string.

        Returns a string containing:
        - Summary of old turns (if any)
        - Recent turns verbatim

        Args:
            history: List of LangChain messages (HumanMessage/AIMessage) or dicts
            llm: LLM instance for summarization (optional, returns raw if None)

        Returns:
            Formatted context string for injection into prompts
        """
        from langchain_core.messages import HumanMessage, AIMessage

        if not history:
            return ""

        total = len(history)

        # If history is short enough, return all verbatim
        if total <= self.max_recent_turns:
            return self._format_history(history)

        # Split into old (to summarize) and recent (verbatim)
        old_turns = history[:-self.max_recent_turns]
        recent_turns = history[-self.max_recent_turns:]

        # Get or create summary for old turns
        summary = self._get_or_create_summary(old_turns, llm)

        # Build combined context
        parts = []
        if summary:
            parts.append(f"[Tóm tắt hội thoại trước đó]\n{summary}")

        if recent_turns:
            recent_str = self._format_history(recent_turns)
            parts.append(f"[Hội thoại gần đây]\n{recent_str}")

        return "\n\n".join(parts)

    def _get_or_create_summary(
        self,
        old_turns: list,
        llm=None,
    ) -> str:
        """Get cached summary or create a new one."""
        turn_count = len(old_turns)

        # Use cached summary if it covers the same turns
        if (
            self._cached_summary
            and self._summary_turn_count == turn_count
        ):
            return self._cached_summary

        # No LLM available — just truncate
        if llm is None:
            return self._truncate_summary(old_turns)

        # Generate new summary
        try:
            summary = self.summarize_old_turns(old_turns, llm)
            self._cached_summary = summary
            self._summary_turn_count = turn_count
            return summary
        except Exception as e:
            logger.warning(f"Summarization failed: {e}. Using truncated fallback.")
            return self._truncate_summary(old_turns)

    def summarize_old_turns(self, turns: list, llm) -> str:
        """
        Summarize old conversation turns using LLM.

        Args:
            turns: List of messages to summarize
            llm: LLM instance

        Returns:
            Short summary paragraph (2-3 sentences)
        """
        conversation_str = self._format_history(turns)

        # Truncate if too long for LLM
        if len(conversation_str) > 4000:
            conversation_str = conversation_str[:4000] + "\n... (truncated)"

        prompt = SUMMARIZE_PROMPT.format(conversation=conversation_str)
        response = llm.invoke(prompt).content.strip()

        logger.info(f"Summarized {len(turns)} turns into {len(response)} chars")
        return response

    def invalidate_cache(self):
        """Clear cached summary (call when history changes significantly)."""
        self._cached_summary = None
        self._summary_turn_count = 0

    @staticmethod
    def _format_history(history: list) -> str:
        """Format history list into readable text."""
        from langchain_core.messages import HumanMessage, AIMessage

        lines = []
        for m in history:
            if isinstance(m, HumanMessage):
                lines.append(f"User: {m.content}")
            elif isinstance(m, AIMessage):
                lines.append(f"AI: {m.content}")
            elif isinstance(m, dict):
                role = "User" if m.get("role") == "user" else "AI"
                lines.append(f"{role}: {m.get('content', '')}")

        return "\n".join(lines)

    @staticmethod
    def _truncate_summary(turns: list, max_chars: int = 500) -> str:
        """Fallback: truncate old turns without LLM summarization."""
        text = ConversationMemory._format_history(turns)
        if len(text) > max_chars:
            return text[:max_chars] + "... (truncated)"
        return text
