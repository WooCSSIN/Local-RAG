"""
evaluate.py — RAGAS 0.2 evaluation pipeline
Dùng model local (Ollama) thay vì OpenAI để evaluate — hoàn toàn offline.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_evaluation(
    test_cases: list[dict],
    llm_model: str = "qwen2.5:7b",
    ollama_url: str = "http://localhost:11434",
    output_path: Optional[str] = None,
) -> dict:
    """
    Chạy RAGAS evaluation với local Ollama LLM.

    Args:
        test_cases: List of dicts with keys:
            - question (str)
            - answer (str)  — model output
            - contexts (list[str])  — retrieved chunks
            - ground_truth (str)  — expected answer
        llm_model: Tên Ollama model dùng để evaluate
        ollama_url: URL Ollama server
        output_path: Nếu có, lưu kết quả CSV ra file này

    Returns:
        dict với các metric scores

    Example:
        test_cases = [
            {
                "question": "RAG là gì?",
                "answer": "RAG là Retrieval-Augmented Generation...",
                "contexts": ["RAG kết hợp tìm kiếm với sinh văn bản..."],
                "ground_truth": "RAG là kỹ thuật kết hợp retrieval và generation",
            }
        ]
        results = run_evaluation(test_cases)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            f"Thiếu dependency: {e}\n"
            "Chạy: pip install ragas langchain-ollama datasets"
        )

    if not test_cases:
        raise ValueError("test_cases không được rỗng")

    # Validate keys
    required_keys = {"question", "answer", "contexts"}
    for i, case in enumerate(test_cases):
        missing = required_keys - set(case.keys())
        if missing:
            raise ValueError(f"test_cases[{i}] thiếu keys: {missing}")

    logger.info(f"Evaluating {len(test_cases)} test cases với model: {llm_model}")

    # Dùng Ollama local thay OpenAI
    eval_llm = LangchainLLMWrapper(
        ChatOllama(model=llm_model, base_url=ollama_url, temperature=0)
    )
    eval_emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    )

    dataset = Dataset.from_list(test_cases)

    # Chọn metrics tùy theo có ground_truth không
    has_ground_truth = all("ground_truth" in c for c in test_cases)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_ground_truth:
        metrics.append(context_recall)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_emb,
    )

    logger.info(f"Kết quả RAGAS:\n{results}")

    # Lưu CSV nếu cần
    if output_path:
        df = results.to_pandas()
        df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu kết quả → {output_path}")

    return results


def generate_test_cases_from_rag(
    questions: list[str],
    ground_truths: list[str],
    graph,
) -> list[dict]:
    """
    Tạo test cases bằng cách chạy RAG graph trên danh sách câu hỏi.

    Args:
        questions: Danh sách câu hỏi
        ground_truths: Đáp án đúng tương ứng
        graph: LangGraph compiled graph
    Returns:
        List of test case dicts
    """
    assert len(questions) == len(ground_truths), "questions và ground_truths phải cùng độ dài"

    test_cases = []
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        logger.info(f"[{i+1}/{len(questions)}] Generating: {q[:50]}...")
        try:
            result = graph.invoke({
                "question": q,
                "chat_history": [],
                "context": [],
                "answer": "",
                "source_pages": [],
                "needs_retrieval": True,
            })
            test_cases.append({
                "question": q,
                "answer": result["answer"],
                "contexts": [d.page_content for d in result.get("context", [])],
                "ground_truth": gt,
            })
        except Exception as e:
            logger.error(f"Lỗi câu hỏi {i}: {e}")

    return test_cases


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import argparse

    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="RAGAS Evaluation")
    parser.add_argument("--input", required=True, help="File JSON chứa test cases")
    parser.add_argument("--output", default="./metrics/eval_results.csv", help="Output CSV path")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model")
    args = parser.parse_args()

    with open(args.input) as f:
        test_cases = json.load(f)

    results = run_evaluation(
        test_cases=test_cases,
        llm_model=args.model,
        output_path=args.output,
    )
    print(f"\n✅ Evaluation xong. Kết quả:\n{results}")
