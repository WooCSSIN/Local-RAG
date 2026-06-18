"""
setup_v2.py — Kiểm tra và hướng dẫn cài đặt môi trường
Chạy: python setup_v2.py
"""
import sys
import subprocess
import platform


def run(cmd: str) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def header(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def check_python():
    v = sys.version_info
    status = "✅" if v >= (3, 10) else "❌ cần >= 3.10"
    print(f"Python {v.major}.{v.minor}.{v.micro}  {status}")
    return v >= (3, 10)


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA GPU : ✅ {name} ({vram:.1f} GB VRAM)")
        else:
            print("CUDA GPU : ⚠️  Không có GPU CUDA — dùng CPU (chậm hơn)")
    except ImportError:
        print("CUDA GPU : ⚠️  PyTorch chưa cài — LLM chạy qua Ollama nên không bắt buộc")


def check_ollama() -> bool:
    code, out = run("ollama --version")
    if code == 0:
        print(f"Ollama   : ✅ {out}")
        code2, out2 = run("ollama list")
        if code2 == 0:
            lines = [l for l in out2.splitlines() if l.strip() and "NAME" not in l]
            if lines:
                print(f"  Models : {len(lines)} model(s) đã cài")
                for l in lines[:5]:
                    print(f"    • {l.split()[0]}")
            else:
                print("  Models : ⚠️  Chưa có model nào")
                print("  Gợi ý  : ollama pull qwen2.5:7b")
        return True
    else:
        print("Ollama   : ❌ Chưa cài")
        if platform.system() == "Windows":
            print("  → Tải: https://ollama.com/download/windows")
            print("  → Cài xong chạy: ollama pull qwen2.5:7b")
        else:
            print("  → curl -fsSL https://ollama.com/install.sh | sh")
        return False


def check_packages() -> list[str]:
    checks = [
        ("langchain",          "langchain"),
        ("langchain_community","langchain-community"),
        ("langchain_ollama",   "langchain-ollama"),
        ("langgraph",          "langgraph"),
        ("faiss",              "faiss-cpu"),
        ("fastembed",          "fastembed"),
        ("rank_bm25",          "rank-bm25"),
        ("flashrank",          "flashrank"),
        ("pypdf",              "pypdf"),
        ("fitz",               "PyMuPDF"),
        ("gradio",             "gradio"),
        ("ragas",              "ragas"),
        ("pydantic_settings",  "pydantic-settings"),
        ("dotenv",             "python-dotenv"),
    ]

    missing = []
    ok_count = 0
    for imp, pkg in checks:
        try:
            mod = __import__(imp)
            ver = getattr(mod, "__version__", "")
            print(f"  ✅ {pkg:<30} {ver}")
            ok_count += 1
        except ImportError:
            print(f"  ❌ {pkg}")
            missing.append(pkg)

    # Docling — optional
    try:
        import docling  # noqa
        print(f"  ✅ docling (optional)")
    except ImportError:
        print(f"  ⚠️  docling (optional — Word/Excel/HTML support)")

    print(f"\n  {ok_count}/{len(checks)} packages OK")
    return missing


def check_env_file():
    from pathlib import Path
    if Path(".env").exists():
        print(".env      : ✅ Tồn tại")
    elif Path(".env.example").exists():
        import shutil
        shutil.copy(".env.example", ".env")
        print(".env      : ✅ Tạo từ .env.example")
    else:
        print(".env      : ⚠️  Không tìm thấy — dùng cấu hình mặc định")


def check_index():
    from pathlib import Path
    docs_file = Path("qdrant_data/documents.pkl")
    if docs_file.exists():
        import pickle
        try:
            with open(docs_file, "rb") as f:
                docs = pickle.load(f)
            print(f"Index     : ✅ {len(docs)} docs trong vector store")
        except Exception:
            print("Index     : ⚠️  Không đọc được index file")
    else:
        print("Index     : ℹ️  Chưa có index — upload tài liệu sau khi chạy app")


def print_quickstart():
    print("""
╔══════════════════════════════════════════════════╗
║           HƯỚNG DẪN CHẠY NHANH                  ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  1. Cài Ollama (nếu chưa có):                    ║
║     https://ollama.com/download/windows          ║
║                                                  ║
║  2. Pull model LLM:                              ║
║     ollama pull qwen2.5:7b    (5GB VRAM)         ║
║     ollama pull llama3.2:3b   (2GB VRAM, nhẹ)    ║
║     ollama pull phi4-mini     (3.8GB VRAM)       ║
║                                                  ║
║  3. Khởi động Ollama server:                     ║
║     ollama serve                                 ║
║                                                  ║
║  4. Chạy app:                                    ║
║     python app.py                                ║
║                                                  ║
║  5. Mở trình duyệt:                              ║
║     http://localhost:7860                        ║
║                                                  ║
╚══════════════════════════════════════════════════╝
""")


def main():
    header("Local RAG Chatbot Gree — Setup Check")

    print("\n[ Python ]")
    check_python()

    print("\n[ GPU ]")
    check_cuda()

    print("\n[ Ollama ]")
    ollama_ok = check_ollama()

    print("\n[ Packages ]")
    missing = check_packages()

    print("\n[ Config & Index ]")
    check_env_file()
    check_index()

    if missing:
        print(f"\n⚠️  Còn {len(missing)} packages chưa cài:")
        print(f"   pip install {' '.join(missing)}")
        ans = input("\nCài đặt ngay? (y/n): ").strip().lower()
        if ans == "y":
            code, out = run(f"{sys.executable} -m pip install {' '.join(missing)} --only-binary=:all:")
            print(out[-500:] if len(out) > 500 else out)

    print_quickstart()


if __name__ == "__main__":
    main()
