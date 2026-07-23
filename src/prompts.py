"""
prompts.py — Centralized prompt templates for the entire RAG pipeline.
Supports bilingual Vietnamese-English prompts.
All prompts are Python string constants — no Jinja2 dependency needed.
"""

# ------------------------------------------------------------------
# System prompt — injected as SystemMessage in every LLM call
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Bạn là trợ lý thông minh chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp. "
    "Luôn trả lời bằng ngôn ngữ của câu hỏi (tiếng Việt hoặc tiếng Anh). "
    "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ thay vì đoán. "
    "Khi trích dẫn thông tin, hãy đề cập nguồn tài liệu."
)

# ------------------------------------------------------------------
# Node 1: Route — decide whether retrieval is needed
# ------------------------------------------------------------------
ROUTE_PROMPT = """\
Xác định xem câu hỏi dưới đây có cần tra cứu tài liệu không.
Trả lời chỉ với YES hoặc NO.

- YES: hỏi về thông tin cụ thể, nội dung tài liệu, dữ liệu
- NO: chào hỏi, cảm ơn, câu hỏi chung không liên quan tài liệu

{history_section}Câu hỏi: {question}
Cần retrieval? (YES/NO):"""

# ------------------------------------------------------------------
# Node 2: Rewrite — make follow-up questions self-contained
# ------------------------------------------------------------------
REWRITE_PROMPT = """\
Dựa vào lịch sử hội thoại, viết lại câu hỏi thành câu độc lập, đầy đủ nghĩa.
CHỈ trả về câu hỏi đã viết lại, không giải thích thêm.

Lịch sử:
{history}

Câu hỏi gốc: {question}
Câu hỏi viết lại:"""

# ------------------------------------------------------------------
# Node 3: Generate — produce answer from context + history
# ------------------------------------------------------------------
GENERATE_WITH_CONTEXT_PROMPT = """\
Bạn là trợ lý thông minh. Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
Nếu ngữ cảnh không đủ thông tin, hãy nói rõ.
Trả lời bằng ngôn ngữ của câu hỏi.

=== NGỮ CẢNH ===
{context}
================

{history_section}Câu hỏi: {question}

Trả lời:"""

GENERATE_NO_CONTEXT_PROMPT = """\
{history_section}Câu hỏi: {question}
Trả lời:"""

# ------------------------------------------------------------------
# Agent: Query Decomposition
# ------------------------------------------------------------------
DECOMPOSE_CHECK_PROMPT = """\
Phân tích câu hỏi sau và xác định xem nó có cần được tách thành nhiều câu hỏi nhỏ không.

Một câu hỏi cần tách khi:
- Hỏi so sánh giữa 2+ khái niệm
- Hỏi nhiều khía cạnh khác nhau của cùng một chủ đề
- Câu hỏi phức tạp yêu cầu thông tin từ nhiều phần khác nhau của tài liệu

Trả lời chỉ YES hoặc NO.

Câu hỏi: {question}
Cần tách câu hỏi? (YES/NO):"""

DECOMPOSE_PROMPT = """\
Tách câu hỏi phức tạp sau thành 2-4 câu hỏi nhỏ hơn, mỗi câu hỏi tập trung vào một khía cạnh cụ thể.
Trả về mỗi câu hỏi trên một dòng riêng, không giải thích thêm.

Ví dụ:
Câu hỏi: "So sánh ưu nhược điểm của RAG và fine-tuning"
Kết quả:
RAG có những ưu điểm gì?
RAG có những nhược điểm gì?
Fine-tuning có những ưu điểm gì?
Fine-tuning có những nhược điểm gì?

Câu hỏi: {question}
Kết quả:"""

# ------------------------------------------------------------------
# Agent: Answer Grading / Self-Reflection
# ------------------------------------------------------------------
GRADE_PROMPT = """\
Đánh giá chất lượng câu trả lời dựa trên 3 tiêu chí:
1. Faithfulness (Trung thực): Câu trả lời có dựa trên ngữ cảnh được cung cấp không?
2. Relevancy (Liên quan): Câu trả lời có trả lời đúng câu hỏi không?
3. Completeness (Đầy đủ): Câu trả lời có đủ thông tin cần thiết không?

Trả lời theo format JSON:
{{"faithfulness": "PASS" hoặc "FAIL", "relevancy": "PASS" hoặc "FAIL", "completeness": "PASS" hoặc "FAIL", "overall": "PASS" hoặc "FAIL", "reason": "lý do ngắn gọn"}}

Câu hỏi: {question}

Ngữ cảnh:
{context}

Câu trả lời:
{answer}

Đánh giá:"""

# ------------------------------------------------------------------
# Agent: Context Assessment (Multi-Step Retrieval)
# ------------------------------------------------------------------
ASSESS_CONTEXT_PROMPT = """\
Đánh giá xem ngữ cảnh được cung cấp đã đủ thông tin để trả lời câu hỏi chưa.

Trả lời chỉ với:
- SUFFICIENT: ngữ cảnh đã đủ thông tin
- INSUFFICIENT: ngữ cảnh thiếu thông tin, cần tìm thêm

Câu hỏi: {question}

Ngữ cảnh:
{context}

Đánh giá (SUFFICIENT/INSUFFICIENT):"""

REFINE_QUERY_PROMPT = """\
Ngữ cảnh hiện tại chưa đủ để trả lời câu hỏi. Hãy viết lại câu hỏi theo cách khác để tìm kiếm thông tin tốt hơn.
CHỈ trả về câu hỏi đã viết lại, không giải thích.

Câu hỏi gốc: {question}

Ngữ cảnh hiện có (tóm tắt):
{context_summary}

Câu hỏi viết lại:"""

# ------------------------------------------------------------------
# Agent: Tool Selection
# ------------------------------------------------------------------
TOOL_SELECTION_PROMPT = """\
Xác định xem có cần dùng công cụ bên ngoài để trả lời câu hỏi không, vì tài liệu local không đủ thông tin.

Các công cụ có sẵn:
- web_search: tìm kiếm trên web (cho thông tin mới, sự kiện gần đây)
- none: không cần công cụ (tài liệu đã đủ hoặc câu hỏi không cần)

Trả lời chỉ tên công cụ: web_search hoặc none

Câu hỏi: {question}

Ngữ cảnh hiện có:
{context_summary}

Công cụ cần dùng:"""

# ------------------------------------------------------------------
# Memory: Conversation Summarization
# ------------------------------------------------------------------
SUMMARIZE_PROMPT = """\
Tóm tắt cuộc hội thoại sau thành một đoạn văn ngắn (2-3 câu), giữ lại các thông tin quan trọng:

{conversation}

Tóm tắt:"""

# ------------------------------------------------------------------
# Feature: Mock Interview Mode
# ------------------------------------------------------------------
MOCK_INTERVIEW_SYSTEM_PROMPT = (
    "Bạn là một interviewer kỹ thuật giàu kinh nghiệm. "
    "Nhiệm vụ của bạn là đặt câu hỏi phỏng vấn chất lượng cao, "
    "đánh giá câu trả lởi của ứng viên một cách công bằng, "
    "và đưa ra phản hồi giúp ứng viên cải thiện. "
    "Luôn trả lởi bằng tiếng Việt hoặc tiếng Anh theo ngôn ngữ của câu hỏi."
)

GENERATE_INTERVIEW_QUESTIONS_PROMPT = """\
Dựa trên tài liệu và chủ đề được cung cấp, hãy tạo {num_questions} câu hỏi phỏng vấn ở mức độ {difficulty}.
Mỗi câu hỏi nên kiểm tra hiểu biết sâu, khả năng phân tích hoặc giải quyết vấn đề.

Chủ đề / Vị trí: {topic}
Mô tả bổ sung: {description}

=== TÀI LIỆU THAM KHẢO ===
{context}
==========================

Trả về JSON array theo định dạng sau:
[
  {{"id": 1, "question": "...", "category": "...", "key_points": ["...", "..."]}},
  ...
]

Câu hỏi:"""

EVALUATE_INTERVIEW_ANSWER_PROMPT = """\
Bạn là interviewer đánh giá câu trả lởi của ứng viên.

Câu hỏi:
{question}

Câu trả lởi của ứng viên:
{answer}

Các điểm then chốt cần có:
{key_points}

=== TÀI LIỆU THAM KHẢO ===
{context}
==========================

Hãy đánh giá theo format JSON:
{{
  "score": <số từ 0-10>,
  "strengths": ["điểm mạnh 1", "điểm mạnh 2"],
  "weaknesses": ["điểm yếu 1", "điểm yếu 2"],
  "missing_points": ["nội dung còn thiếu 1", "nội dung còn thiếu 2"],
  "suggested_answer": "câu trả lởi mẫu ngắn gọn, đầy đủ",
  "feedback": "phản hồi tổng quan 2-3 câu"
}}

Đánh giá:"""

FINAL_INTERVIEW_FEEDBACK_PROMPT = """\
Dựa trên toàn bộ buổi phỏng vấn, hãy đưa ra nhận xét tổng kết.

Thông tin buổi phỏng vấn:
{interview_summary}

Hãy đưa ra:
1. Điểm trung bình
2. Điểm mạnh nổi bật
3. Điểm cần cải thiện
4. Lởi khuyên ôn tập cụ thể

Trả lởi bằng tiếng Việt, ngắn gọn, dễ hiểu.

Nhận xét:"""

# ------------------------------------------------------------------
# Helper: Build context string from documents
# ------------------------------------------------------------------
def build_context_string(docs: list) -> str:
    """Format retrieved documents into context string for prompts."""
    if not docs:
        return ""
    parts = []
    for doc in docs:
        filename = doc.metadata.get("filename", "tài liệu")
        page = doc.metadata.get("page")
        source = f"[Nguồn: {filename}"
        if page is not None:
            source += f", trang {page}"
        source += f"]\n{doc.page_content}"
        parts.append(source)
    return "\n\n---\n\n".join(parts)


def build_history_string(history: list, max_turns: int = 6) -> str:
    """Format conversation history for prompts. Returns empty string if no history."""
    from langchain_core.messages import HumanMessage, AIMessage

    if not history:
        return ""

    recent = history[-max_turns:]
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"AI: {m.content}")
        elif isinstance(m, dict):
            role = "User" if m.get("role") == "user" else "AI"
            lines.append(f"{role}: {m.get('content', '')}")

    if not lines:
        return ""
    return "=== LỊCH SỬ ===\n" + "\n".join(lines) + "\n================\n\n"
