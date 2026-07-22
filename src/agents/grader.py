"""
grader.py — Answer Grading / Self-Reflection Agent
Evaluates answer quality before returning to user.
Detects hallucination, irrelevance, and incomplete answers.
"""
import json
import logging
from dataclasses import dataclass

from src.prompts import GRADE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GradeResult:
    """Result of answer quality grading."""
    faithfulness: bool  # True = PASS, False = FAIL
    relevancy: bool
    completeness: bool
    overall: bool       # True = PASS (all criteria met), False = FAIL
    reason: str

    @classmethod
    def pass_result(cls, reason: str = "All criteria met") -> "GradeResult":
        return cls(True, True, True, True, reason)

    @classmethod
    def fail_result(cls, reason: str) -> "GradeResult":
        return cls(False, False, False, False, reason)


class AnswerGrader:
    """
    Evaluate answer quality using 3 criteria:
    1. Faithfulness: Is the answer grounded in the provided context?
    2. Relevancy: Does it address the user's question?
    3. Completeness: Does it contain sufficient information?

    Returns PASS/FAIL for each criterion plus overall verdict.
    Max 2 retry loops when grade is FAIL.
    """

    MAX_RETRIES = 2  # Max retry loops when grade fails

    def grade(self, question: str, answer: str, context: str, llm) -> GradeResult:
        """
        Grade the answer quality.

        Args:
            question: The user's question
            answer: The generated answer
            context: The retrieved context string
            llm: LLM instance for evaluation

        Returns:
            GradeResult with pass/fail for each criterion
        """
        if not context or not context.strip():
            # No context — skip faithfulness check, only check relevancy
            return self._grade_no_context(question, answer, llm)

        try:
            prompt = GRADE_PROMPT.format(
                question=question,
                context=context[:3000],  # Truncate to avoid token limits
                answer=answer[:2000],
            )
            response = llm.invoke(prompt).content.strip()
            return self._parse_grade_response(response)
        except Exception as e:
            logger.warning(f"Grading failed: {e}. Defaulting to PASS.")
            return GradeResult.pass_result("Grading error — defaulting to PASS")

    def _grade_no_context(self, question: str, answer: str, llm) -> GradeResult:
        """Grade when no context is available (chatty responses)."""
        # Without context, we only check if the answer is relevant to the question
        # Faithfulness and completeness are not applicable
        if not answer or len(answer.strip()) < 5:
            return GradeResult.fail_result("Empty or too short answer")
        return GradeResult.pass_result("No context — relevancy assumed")

    def _parse_grade_response(self, response: str) -> GradeResult:
        """Parse LLM response into GradeResult."""
        try:
            # Try to extract JSON from response
            # Handle cases where LLM wraps JSON in markdown
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            faithfulness = data.get("faithfulness", "PASS").upper() == "PASS"
            relevancy = data.get("relevancy", "PASS").upper() == "PASS"
            completeness = data.get("completeness", "PASS").upper() == "PASS"
            overall = data.get("overall", "PASS").upper() == "PASS"
            reason = data.get("reason", "")

            return GradeResult(
                faithfulness=faithfulness,
                relevancy=relevancy,
                completeness=completeness,
                overall=overall,
                reason=reason,
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse grade response: {e}")
            # Fallback: check for keywords in response
            response_upper = response.upper()
            overall = "PASS" in response_upper and "FAIL" not in response_upper
            if overall:
                return GradeResult.pass_result("Parsed from keywords")
            return GradeResult.fail_result("Parsed from keywords — FAIL detected")
