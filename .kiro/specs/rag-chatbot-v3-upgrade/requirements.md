# Requirements Document

## Introduction

Đây là tài liệu yêu cầu cho việc nâng cấp **Local RAG Chatbot v2 lên v3**. Hệ thống hiện tại sử dụng stack: Groq/Ollama/OpenAI (LLM), nomic-embed-text-v1.5 (FastEmbed), FAISS in-memory + BM25 hybrid search, FlashRank reranker, Gradio 6.x UI và LangGraph orchestration. Bản nâng cấp v3 tập trung vào 5 nhóm cải tiến: hiệu năng khởi động, trải nghiệm người dùng, chất lượng câu trả lời, lưu trữ phiên hội thoại và mở rộng định dạng tài liệu.

---

## Glossary

- **Chatbot**: Ứng dụng RAG Chatbot Local chạy trên Gradio.
- **Index**: Cấu trúc lưu trữ vector (FAISS) và keyword (BM25) để tìm kiếm tài liệu.
- **FAISS_Index**: Chỉ mục vector FAISS được lưu xuống đĩa dưới dạng file nhị phân.
- **DocumentProcessor**: Module xử lý và chunk tài liệu từ nhiều định dạng.
- **Retriever**: Module `HybridRAGRetriever` thực hiện hybrid search và reranking.
- **LLM_Factory**: Module `llm_factory.py` tạo LLM instance theo provider được cấu hình.
- **LLM_Cache**: Cơ chế cache LLM instance để tránh khởi tạo lại mỗi lần chat.
- **Session**: Một phiên hội thoại độc lập với lịch sử riêng biệt.
- **Session_Store**: Nơi lưu trữ các Session trên đĩa theo định dạng JSON.
- **Chat_History**: Danh sách các lượt hội thoại (user/assistant) trong một Session.
- **Duplicate_Guard**: Cơ chế kiểm tra file đã được index trước đó chưa dựa trên tên file và hash nội dung.
- **Progress_Bar**: Thanh tiến trình hiển thị trạng thái xử lý file trong UI Gradio.
- **Export**: Chức năng xuất Chat_History ra file văn bản (TXT hoặc Markdown).
- **System_Prompt**: Văn bản định nghĩa vai trò và hành vi của LLM ở đầu mỗi cuộc trò chuyện.
- **Docling**: Thư viện xử lý tài liệu hỗ trợ PDF, DOCX, XLSX, HTML, Markdown với cấu trúc cao.
- **OCR**: Nhận dạng ký tự quang học (Optical Character Recognition) để trích xuất văn bản từ ảnh.
- **Streaming**: Cơ chế trả về câu trả lời từng phần (token-by-token) từ LLM.

---

## Requirements

---

### Requirement 1: Lưu Persistent FAISS Index

**User Story:** As a developer, I want the FAISS index to be saved to and loaded from disk automatically, so that the chatbot starts in under 5 seconds instead of rebuilding the index every time.

#### Acceptance Criteria

1. WHEN the Retriever successfully builds a FAISS index after adding new documents, THE Retriever SHALL serialize the FAISS index to disk at the path configured by `qdrant_path/faiss_index.bin`.
2. WHEN the Retriever initializes and a valid FAISS index file exists on disk, THE Retriever SHALL load the FAISS index from disk instead of rebuilding it from documents.
3. WHEN the Retriever loads a FAISS index from disk, THE Retriever SHALL verify that the number of vectors in the loaded index matches the number of documents in `documents.pkl`.
4. IF the FAISS index file on disk is corrupted or does not match the document count, THEN THE Retriever SHALL delete the invalid index file and rebuild the FAISS index from the existing documents.
5. WHEN the Chatbot starts with a pre-existing FAISS index on disk, THE Chatbot SHALL complete initialization within 5 seconds for up to 300 documents.
6. WHEN documents are cleared via the "Xóa index" function, THE Retriever SHALL delete both `documents.pkl` and `faiss_index.bin` from disk.
7. WHEN the FAISS index file is determined to be corrupted or invalid, THE Retriever SHALL delete `faiss_index.bin` independently of `documents.pkl` to allow reconstruction without affecting the document store.

---

### Requirement 2: Cache LLM Instance

**User Story:** As a developer, I want the LLM instance to be created once and reused across all chat turns, so that each chat request does not incur the overhead of re-instantiating the LLM client.

#### Acceptance Criteria

1. THE LLM_Factory SHALL cache the LLM instance after the first successful call to `get_llm(config)`.
2. WHEN `get_llm(config)` is called again with the same provider and model configuration, THE LLM_Factory SHALL return the cached instance without creating a new one.
3. WHEN the LLM provider or model name changes in the configuration at runtime, THE LLM_Factory SHALL invalidate the cache and create a new LLM instance.
4. WHEN the cached LLM instance raises a connection error during a chat request, THE LLM_Factory SHALL clear the cache and attempt to create a new instance as the first priority, then apply any pending configuration changes to the newly created instance before retrying once.

---

### Requirement 3: Progress Bar khi Upload và Index

**User Story:** As a user, I want to see a progress bar while files are being uploaded and indexed, so that I know the system is working and approximately how long it will take.

#### Acceptance Criteria

1. WHEN a user clicks "Xử lý & Index" with one or more files selected, THE Chatbot SHALL display a progress indicator showing the current file being processed and the total number of files (e.g., "Đang xử lý 2/3: document.pdf").
2. WHEN each file finishes processing, THE Chatbot SHALL update the progress indicator to reflect the completed count.
3. WHEN all files have been processed, THE Chatbot SHALL replace the progress indicator with a final summary showing the number of chunks added and the total document count in the index.
4. IF a file fails to process, THEN THE Chatbot SHALL display the error message for that file in the status area without stopping the processing of remaining files.

---

### Requirement 4: Ngăn Chặn Duplicate Documents

**User Story:** As a user, I want the system to prevent me from indexing the same file twice, so that my search results are not polluted with duplicate content.

#### Acceptance Criteria

1. WHEN a user uploads a file for indexing, THE Duplicate_Guard SHALL compute a SHA-256 hash of the file content.
2. THE Retriever SHALL maintain a set of hashes of all previously indexed files in the persistent store.
3. WHEN a file's SHA-256 hash matches a hash already present in the persistent store, THE Chatbot SHALL skip indexing that file and display a warning message: "File '[tên file]' đã được index trước đó. Bỏ qua."
4. WHEN a file's SHA-256 hash is new, THE Retriever SHALL add the file's hash to the persistent store after successful indexing.
5. WHEN the index is cleared, THE Retriever SHALL also clear the set of stored file hashes.

---

### Requirement 5: Xuất Chat Ra File

**User Story:** As a user, I want to export my chat history to a file, so that I can save and share conversation results.

#### Acceptance Criteria

1. THE Chatbot SHALL display an "Xuất chat" button in the chat interface at all times.
2. WHEN a user clicks "Xuất chat" and the Chat_History is not empty, THE Chatbot SHALL generate a downloadable file containing the full Chat_History.
3. THE Export file SHALL be available in both TXT and Markdown formats, selectable by the user before downloading.
4. WHEN generating a TXT export, THE Chatbot SHALL format each turn as `[HH:MM:SS] User: <nội dung>` and `[HH:MM:SS] Assistant: <nội dung>` with timestamps derived from when each message was received.
5. WHEN generating a Markdown export, THE Chatbot SHALL format each user message as a `**User:**` bold heading and each assistant message as a plain paragraph, with a horizontal rule between turns.
6. WHEN a user selects TXT format, THE Chatbot SHALL apply only TXT formatting; WHEN a user selects Markdown format, THE Chatbot SHALL apply only Markdown formatting, ensuring only the selected format is processed and applied.
7. WHEN a user clicks "Xuất chat" and the Chat_History is empty, THE Chatbot SHALL display a notification: "Chưa có nội dung chat để xuất."

---

### Requirement 6: Cải Thiện System Prompt Tiếng Việt

**User Story:** As a developer, I want the system prompt to be written with proper Vietnamese diacritics and to include a dedicated system role, so that the LLM produces more accurate and natural Vietnamese responses.

#### Acceptance Criteria

1. THE Chatbot SHALL include all Vietnamese text in source code with full Unicode diacritics (e.g., "Bạn là trợ lý thông minh" not "Ban la tro ly thong minh").
2. WHEN building a prompt for the LLM, THE Chatbot SHALL include a dedicated system role message that defines the assistant's persona, language preference, and instruction to cite sources.
3. THE System_Prompt SHALL instruct the LLM to respond in the same language as the user's question.
4. THE System_Prompt SHALL instruct the LLM to state clearly when the provided context does not contain sufficient information to answer a question.
5. THE Chatbot SHALL pass the system instruction as a `SystemMessage` and the user query as a `HumanMessage` using LangChain message types for all providers (Groq, OpenAI, Ollama), regardless of whether the underlying provider natively distinguishes between system and user roles.

---

### Requirement 7: Ổn Định Streaming

**User Story:** As a user, I want the streaming response to be stable and never show a broken or incomplete message, so that I can read the answer as it is being generated without errors.

#### Acceptance Criteria

1. WHEN the LLM streams a response, THE Chatbot SHALL accumulate each token chunk and yield the complete accumulated text to the Gradio UI after each chunk.
2. IF a chunk received from the LLM has no `content` attribute or contains an empty string, THEN THE Chatbot SHALL skip that chunk without raising an exception.
3. IF the LLM stream raises an exception mid-stream, THEN THE Chatbot SHALL stop streaming, append an error notice to the accumulated text, and yield the final text to the UI.
4. WHEN streaming is complete, THE Chatbot SHALL append source citations (if any) to the final message in a single update.

---

### Requirement 8: Lưu Trữ Lịch Sử Chat Persistent

**User Story:** As a user, I want my chat history to be saved to disk so that it is not lost when I close and reopen the application.

#### Acceptance Criteria

1. WHEN a chat turn is completed (both user message and assistant response are ready), THE Chatbot SHALL append the turn to the current Session and persist the Session to the Session_Store.
2. THE Session_Store SHALL store sessions as individual JSON files in the directory `./chat_sessions/`, with each file named by session ID.
3. WHEN the Chatbot starts, THE Chatbot SHALL load the most recent Session from the Session_Store and display it in the chat interface.
4. WHEN a user clicks "Xóa chat", THE Chatbot SHALL clear the current Chat_History from the UI and create a new empty Session, without deleting any previously saved sessions.
5. THE Session_Store SHALL store each message with at minimum: role (`user` or `assistant`), content text, and an ISO 8601 UTC timestamp.
6. IF writing to the Session_Store fails due to a disk error, THEN THE Chatbot SHALL log the error, accept the data loss for that turn, and continue operation without crashing or retrying.

---

### Requirement 9: Multi-Session — Chuyển Đổi Giữa Các Phiên Chat

**User Story:** As a user, I want to create and switch between multiple chat sessions, so that I can keep separate conversation threads for different topics or documents.

#### Acceptance Criteria

1. THE Chatbot UI SHALL display a session list panel showing all saved sessions ordered by last-modified time, most recent first.
2. WHEN a user clicks on a session in the session list, THE Chatbot SHALL load that session's Chat_History into the chat interface, replacing the currently displayed history.
3. THE Chatbot UI SHALL provide a "Phiên mới" button that creates a new empty Session with a system-generated name (e.g., "Phiên 2025-07-15 14:30").
4. THE Chatbot UI SHALL allow the user to rename a session by clicking on the session name in the session list panel.
5. WHEN a user renames a session, THE Session_Store SHALL update the session's display name in the corresponding JSON file immediately.
6. THE Chatbot UI SHALL provide a "Xóa phiên" button for each session in the session list, which permanently deletes that session's JSON file from the Session_Store; session deletion SHALL only occur through explicit user action via these buttons and SHALL NOT be triggered automatically by the system.
7. IF a user attempts to delete the currently active session, THEN THE Chatbot SHALL create a new empty Session before deleting the old one, so the chat interface is never left in an undefined state.

---

### Requirement 10: Hỗ Trợ Định Dạng DOCX và XLSX

**User Story:** As a user, I want to upload Word documents and Excel spreadsheets, so that I can ask questions about content in those file formats.

#### Acceptance Criteria

1. THE DocumentProcessor SHALL accept files with extensions `.docx` and `.xlsx` as valid input formats.
2. WHEN processing a `.docx` file, THE DocumentProcessor SHALL extract the full text content using Docling and split it into chunks using the existing `RecursiveCharacterTextSplitter`.
3. WHEN processing an `.xlsx` file, THE DocumentProcessor SHALL extract data from all sheets, convert each row to a text representation, and split the result into chunks.
4. WHEN processing a `.xlsx` file specifically, THE DocumentProcessor SHALL include the sheet name and column headers as context in the chunk metadata to preserve structure; this Excel-specific metadata requirement SHALL NOT apply when processing `.docx` or other non-spreadsheet formats.
5. THE Chatbot UI file upload component SHALL accept `.docx` and `.xlsx` file types in addition to the existing `.pdf`, `.txt`, `.html`, `.md` types.
6. IF Docling is not installed and a `.docx` file is uploaded, THEN THE DocumentProcessor SHALL fall back to the `python-docx` library and log a warning.
7. IF Docling is not installed and a `.xlsx` file is uploaded, THEN THE DocumentProcessor SHALL fall back to the `openpyxl` library and log a warning.

---

### Requirement 11: Hỗ Trợ OCR Cho Ảnh (Nice-to-Have)

**User Story:** As a user, I want to upload image files (PNG, JPG) so that text content in scanned documents or screenshots can be extracted and indexed.

#### Acceptance Criteria

1. WHERE Docling with OCR support is installed, THE DocumentProcessor SHALL accept files with extensions `.png`, `.jpg`, and `.jpeg` as valid input formats.
2. WHERE Docling with OCR support is installed and an image file is provided, THE DocumentProcessor SHALL run OCR on the image to extract text; IF text is obtained by any means, THE DocumentProcessor SHALL attempt to create indexable Document chunks and SHALL fail the entire operation if chunk creation fails.
3. IF Docling with OCR support is not installed and an image file is uploaded, THEN THE Chatbot SHALL display an informative error message: "OCR chưa được kích hoạt. Cài đặt Docling với OCR để hỗ trợ định dạng ảnh."
4. WHEN an image file is processed successfully, THE DocumentProcessor SHALL include the source filename and file type in the chunk metadata.
