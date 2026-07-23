"""
src/mock_interview.py — Mock Interview Mode engine.
Generates interview questions from documents, evaluates answers,
and provides structured feedback for interview preparation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts import (
    MOCK_INTERVIEW_SYSTEM_PROMPT,
    GENERATE_INTERVIEW_QUESTIONS_PROMPT,
    EVALUATE_INTERVIEW_ANSWER_PROMPT,
    FINAL_INTERVIEW_FEEDBACK_PROMPT,
    build_context_string,
)


@dataclass
class InterviewQuestion:
    """Single interview question."""

    id: int
    question: str
    category: str
    key_points: list[str]


@dataclass
class InterviewResult:
    """Result of evaluating one answer."""

    score: float
    strengths: list[str]
    weaknesses: list[str]
    missing_points: list[str]
    suggested_answer: str
    feedback: str


@dataclass
class InterviewSession:
    """State for an ongoing mock interview."""

    topic: str
    difficulty: str
    questions: list[InterviewQuestion] = field(default_factory=list)
    current_index: int = 0
    answers: list[str] = field(default_factory=list)
    results: list[InterviewResult | None] = field(default_factory=list)
    finished: bool = False

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.questions)

    @property
    def average_score(self) -> float:
        scores = [r.score for r in self.results if r is not None]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def to_summary(self) -> str:
        lines = [
            f"Chủ đề: {self.topic}",
            f"Mức độ: {self.difficulty}",
            f"Số câu hỏi: {len(self.questions)}",
            f"Điểm trung bình: {self.average_score:.1f}/10",
        ]
        for i, q in enumerate(self.questions):
            a = self.answers[i] if i < len(self.answers) else "Chưa trả lởi"
            r = self.results[i]
            score = f"{r.score:.1f}/10" if r else "Chưa đánh giá"
            lines.append(f"\nCâu {i + 1}: {q.question}")
            lines.append(f"Trả lởi: {a}")
            lines.append(f"Điểm: {score}")
        return "\n".join(lines)


class MockInterviewEngine:
    """Engine for mock interview sessions backed by an LLM."""

    def __init__(self, llm: Any, retriever: Any | None = None):
        self.llm = llm
        self.retriever = retriever

    def _call_llm(self, prompt: str, system: str | None = None) -> str:
        """Simple LLM call returning string content."""
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract JSON array/object from LLM output."""
        # Try to find JSON block
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find first JSON array/object
        for pattern in [r"\[.*\]", r"\{.*\}"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not extract JSON from: {text[:200]}")

    def generate_questions(
        self,
        topic: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        description: str = "",
    ) -> list[InterviewQuestion]:
        """Generate interview questions based on topic and optional documents."""
        context = ""
        if self.retriever is not None:
            # Retrieve documents related to the topic
            query = f"{topic} {description}".strip()
            try:
                docs = self.retriever.retrieve(query, top_k=8)
                context = build_context_string(docs)
            except Exception:
                context = "Không có tài liệu tham khảo."

        if not context.strip():
            context = "Không có tài liệu tham khảo. Hãy dựa vào kiến thức chung về chủ đề."

        prompt = GENERATE_INTERVIEW_QUESTIONS_PROMPT.format(
            num_questions=num_questions,
            difficulty=difficulty,
            topic=topic,
            description=description or "Không có",
            context=context,
        )
        response = self._call_llm(prompt, system=MOCK_INTERVIEW_SYSTEM_PROMPT)
        data = self._extract_json(response)

        questions: list[InterviewQuestion] = []
        for item in data:
            if isinstance(item, dict):
                questions.append(
                    InterviewQuestion(
                        id=item.get("id", len(questions) + 1),
                        question=item.get("question", ""),
                        category=item.get("category", "General"),
                        key_points=item.get("key_points", []),
                    )
                )
        return questions

    def evaluate_answer(
        self,
        question: InterviewQuestion,
        answer: str,
    ) -> InterviewResult:
        """Evaluate a single answer."""
        context = ""
        if self.retriever is not None:
            try:
                docs = self.retriever.retrieve(question.question, top_k=5)
                context = build_context_string(docs)
            except Exception:
                context = ""

        if not context.strip():
            context = "Không có tài liệu tham khảo."

        prompt = EVALUATE_INTERVIEW_ANSWER_PROMPT.format(
            question=question.question,
            answer=answer,
            key_points="\n".join(f"- {kp}" for kp in question.key_points),
            context=context,
        )
        response = self._call_llm(prompt, system=MOCK_INTERVIEW_SYSTEM_PROMPT)
        data = self._extract_json(response)

        return InterviewResult(
            score=float(data.get("score", 0)),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            missing_points=data.get("missing_points", []),
            suggested_answer=data.get("suggested_answer", ""),
            feedback=data.get("feedback", ""),
        )

    def get_final_feedback(self, session: InterviewSession) -> str:
        """Generate final feedback for the completed session."""
        summary = session.to_summary()
        prompt = FINAL_INTERVIEW_FEEDBACK_PROMPT.format(interview_summary=summary)
        return self._call_llm(prompt, system=MOCK_INTERVIEW_SYSTEM_PROMPT)

    def start_session(
        self,
        topic: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        description: str = "",
    ) -> InterviewSession:
        """Create a new interview session with generated questions."""
        questions = self.generate_questions(topic, num_questions, difficulty, description)
        return InterviewSession(
            topic=topic,
            difficulty=difficulty,
            questions=questions,
            current_index=0,
            answers=[],
            results=[],
        )

    def submit_answer(
        self,
        session: InterviewSession,
        answer: str,
    ) -> InterviewResult:
        """Submit answer for current question and advance."""
        if session.is_complete:
            raise ValueError("Interview session is already complete")

        question = session.questions[session.current_index]
        result = self.evaluate_answer(question, answer)

        session.answers.append(answer)
        session.results.append(result)
        session.current_index += 1

        if session.current_index >= len(session.questions):
            session.finished = True

        return result
