"""
tools.py — External Tool Integration Agent
Provides web search and other tools when local documents are insufficient.
"""
import logging
from dataclasses import dataclass

from src.prompts import TOOL_SELECTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of tool execution."""
    tool_name: str
    success: bool
    content: str
    sources: list[str]  # URLs or references


class ToolAgent:
    """
    Decide when to use external tools and execute them.

    Available tools:
    - web_search: DuckDuckGo search for current/factual information
    - none: No tool needed (local docs sufficient)
    """

    def should_use_tool(self, question: str, context_summary: str, llm) -> str:
        """
        Determine if an external tool is needed.

        Returns:
            Tool name string: "web_search" or "none"
        """
        try:
            prompt = TOOL_SELECTION_PROMPT.format(
                question=question,
                context_summary=context_summary[:1000] if context_summary else "(Không có tài liệu)",
            )
            response = llm.invoke(prompt).content.strip().lower()

            if "web_search" in response:
                logger.info(f"Tool selection: web_search for '{question[:50]}'")
                return "web_search"
            return "none"
        except Exception as e:
            logger.warning(f"Tool selection failed: {e}. Defaulting to none.")
            return "none"

    def web_search(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            ToolResult with search results
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")
            return ToolResult(
                tool_name="web_search",
                success=False,
                content="Web search unavailable: duckduckgo-search package not installed.",
                sources=[],
            )

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return ToolResult(
                    tool_name="web_search",
                    success=True,
                    content="No results found.",
                    sources=[],
                )

            # Format results
            content_parts = []
            sources = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                content_parts.append(f"**{title}**\n{body}")
                if href:
                    sources.append(href)

            content = "\n\n---\n\n".join(content_parts)
            logger.info(f"Web search returned {len(results)} results")
            return ToolResult(
                tool_name="web_search",
                success=True,
                content=content,
                sources=sources,
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolResult(
                tool_name="web_search",
                success=False,
                content=f"Search error: {e}",
                sources=[],
            )

    def execute_tool(self, tool_name: str, query: str) -> ToolResult:
        """
        Execute the specified tool.

        Args:
            tool_name: Name of the tool to execute
            query: Query to pass to the tool

        Returns:
            ToolResult with execution results
        """
        if tool_name == "web_search":
            return self.web_search(query)
        elif tool_name == "none":
            return ToolResult(
                tool_name="none",
                success=True,
                content="",
                sources=[],
            )
        else:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content=f"Unknown tool: {tool_name}",
                sources=[],
            )
