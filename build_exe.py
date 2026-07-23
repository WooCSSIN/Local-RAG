# ==================================================
# Build Local-RAG thành file .exe (Windows)
# Chạy: python build_exe.py
# Output: D:\Local-RAG-Build\dist\Local-RAG
# ==================================================
import os
import sys
import shutil
import PyInstaller.__main__

# Build trên ổ D để tránh đầy ổ C
BUILD_DIR = r"D:\Local-RAG-Build"
WORK_DIR = os.path.join(BUILD_DIR, "build")
DIST_DIR = os.path.join(BUILD_DIR, "dist")
SPEC_DIR = BUILD_DIR

# Dọn dẹp build cũ
if os.path.exists(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)
os.makedirs(WORK_DIR)
os.makedirs(DIST_DIR)

# Lệnh PyInstaller
args = [
    "app.py",                            # Entry point
    "--name=Local-RAG",                  # Tên app
    "--onedir",                          # Một folder thay vì 1 file (khởi động nhanh hơn)
    "--windowed",                        # Không hiện console
    "--noconfirm",                       # Ghi đè output cũ
    f"--workpath={WORK_DIR}",
    f"--distpath={DIST_DIR}",
    f"--specpath={SPEC_DIR}",
    # Thêm các file/folder cần thiết
    "--add-data", f"src{os.pathsep}src",
    "--add-data", f".env.example{os.pathsep}.",
    "--add-data", f"requirements.txt{os.pathsep}.",
    "--add-data", f"requirements-v2.txt{os.pathsep}.",
    "--add-data", f"setup_v2.py{os.pathsep}.",
    "--add-data", f"README.md{os.pathsep}.",
    # Icon nếu có
    # "--icon=assets/icon.ico",
    # Hidden imports
    "--hidden-import", "src.agentic_graph",
    "--hidden-import", "src.agents.decomposer",
    "--hidden-import", "src.agents.grader",
    "--hidden-import", "src.agents.retrieval_agent",
    "--hidden-import", "src.agents.tools",
    "--hidden-import", "src.prompts",
    "--hidden-import", "src.memory",
    "--hidden-import", "src.document_processor",
    "--hidden-import", "src.retriever",
    "--hidden-import", "src.llm_factory",
    "--hidden-import", "src.session_manager",
    "--hidden-import", "src.utils",
    "--hidden-import", "faiss",
    "--hidden-import", "rank_bm25",
    "--hidden-import", "flashrank",
    "--hidden-import", "fastembed",
    "--hidden-import", "duckduckgo_search",
    "--hidden-import", "gradio",
    "--hidden-import", "langgraph",
]

print("Bắt đầu build Local-RAG...")
print(f"Output: {DIST_DIR}")
PyInstaller.__main__.run(args)
print("Build hoàn tất!")
print(f"Folder phần mềm: {DIST_DIR}\\Local-RAG")
print(f"File chạy: {DIST_DIR}\\Local-RAG\\Local-RAG.exe")
