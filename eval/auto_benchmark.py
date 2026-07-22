"""
auto_benchmark.py — Automated RAG pipeline benchmarking
Generates test cases from indexed documents and compares metrics
across different retrieval/generation configurations.

Usage:
    python eval/auto_benchmark.py --questions 10 --model qwen2.5:7b
    python eval/auto_benchmark.py --input questions.json --output results.csv
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def generate_questions_from_docs(
    retriever,
    llm,
    num_questions: int = 10,
) -> list[dict]:
    """
    Auto-generate evaluation questions from indexed documents.

    Args:
        retriever: HybridRAGRetriever with documents
        llm: LLM instance for question generation
        num_questions: Number of questions to generate

    Returns:
        List of dicts with 'question' and 'ground_truth' keys
    """
    if retriever.doc_count == 0:
        raise ValueError("No documents in index. Upload documents first.")

    # Sample documents for question generation
    import random
    sample_size = min(num_questions * 2, retriever.doc_count)
    sample_indices = random.sample(range(retriever.doc_count), sample_size)
    sample_docs = [retriever.documents[i] for i in sample_indices]

    # Build context from sampled docs
    context_parts = []
    for i, doc in enumerate(sample_docs):
        content = doc.page_content[:500]
        source = doc.metadata.get("filename", "unknown")
        context_parts.append(f"[Doc {i+1}: {source}]\n{content}")

    full_context = "\n\n---\n\n".join(context_parts[:num_questions])

    prompt = f"""Based on the following document excerpts, generate {num_questions} diverse questions with their ground truth answers.

Each question should:
1. Be answerable from the given context
2. Test different aspects (factual, comparative, explanatory)
3. Be in the same language as the document content

Context:
{full_context}

Return as JSON array:
[
  {{"question": "...", "ground_truth": "..."}},
  ...
]

Generate {num_questions} question-answer pairs:"""

    try:
        response = llm.invoke(prompt).content.strip()

        # Extract JSON from response
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        pairs = json.loads(json_str.strip())

        # Validate
        valid = []
        for p in pairs:
            if "question" in p and "ground_truth" in p:
                valid.append(p)

        logger.info(f"Generated {len(valid)} valid question-answer pairs")
        return valid[:num_questions]

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return []


def run_benchmark(
    retriever,
    config,
    questions: list[dict],
    output_path: Optional[str] = None,
) -> dict:
    """
    Run RAG benchmark on a set of questions.

    Args:
        retriever: HybridRAGRetriever instance
        config: RAGConfig instance
        questions: List of dicts with 'question' and 'ground_truth'
        output_path: Optional CSV output path

    Returns:
        Dict with benchmark results and metrics
    """
    from src.agentic_graph import build_agentic_graph, make_agentic_state

    graph = build_agentic_graph(retriever, config)

    results = []
    total_time = 0

    for i, qa in enumerate(questions):
        question = qa["question"]
        ground_truth = qa.get("ground_truth", "")

        logger.info(f"[{i+1}/{len(questions)}] Benchmarking: {question[:50]}...")

        start = time.time()
        try:
            state = make_agentic_state(question)
            result = graph.invoke(state)

            elapsed = time.time() - start
            total_time += elapsed

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": result.get("answer", ""),
                "contexts": [d.page_content for d in result.get("context", [])],
                "source_pages": result.get("source_pages", []),
                "retrieval_loops": result.get("retrieval_loop_count", 0),
                "grade_passed": result.get("grade_passed", None),
                "elapsed_seconds": round(elapsed, 2),
            })

            logger.info(f"  Answer: {result.get('answer', '')[:100]}... ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  Error: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": f"ERROR: {e}",
                "contexts": [],
                "source_pages": [],
                "retrieval_loops": 0,
                "grade_passed": None,
                "elapsed_seconds": round(elapsed, 2),
            })

    # Summary statistics
    successful = [r for r in results if not r["answer"].startswith("ERROR:")]
    avg_time = total_time / len(results) if results else 0
    grade_pass_rate = (
        sum(1 for r in successful if r.get("grade_passed")) / len(successful)
        if successful else 0
    )
    avg_contexts = (
        sum(len(r["contexts"]) for r in successful) / len(successful)
        if successful else 0
    )

    summary = {
        "total_questions": len(questions),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "avg_time_seconds": round(avg_time, 2),
        "grade_pass_rate": round(grade_pass_rate, 3),
        "avg_contexts_retrieved": round(avg_contexts, 1),
        "results": results,
    }

    # Save results
    if output_path:
        _save_results(summary, output_path)

    return summary


def _save_results(summary: dict, output_path: str):
    """Save benchmark results to CSV and JSON."""
    output = Path(output_path)

    # Save detailed JSON
    json_path = output.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Detailed results saved: {json_path}")

    # Save CSV summary
    try:
        import pandas as pd
        rows = []
        for r in summary["results"]:
            rows.append({
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "answer": r["answer"][:200],
                "grade_passed": r.get("grade_passed"),
                "contexts_count": len(r["contexts"]),
                "retrieval_loops": r["retrieval_loops"],
                "elapsed_seconds": r["elapsed_seconds"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(output, index=False)
        logger.info(f"CSV summary saved: {output}")
    except ImportError:
        logger.warning("pandas not available, skipping CSV export")


def print_comparison(results_a: dict, results_b: dict, label_a: str = "A", label_b: str = "B"):
    """Print side-by-side comparison of two benchmark runs."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {label_a:>12} {label_b:>12}")
    print(f"  {'-'*50}")
    print(f"  {'Avg Time (s)':<25} {results_a['avg_time_seconds']:>12} {results_b['avg_time_seconds']:>12}")
    print(f"  {'Grade Pass Rate':<25} {results_a['grade_pass_rate']:>12} {results_b['grade_pass_rate']:>12}")
    print(f"  {'Avg Contexts':<25} {results_a['avg_contexts_retrieved']:>12} {results_b['avg_contexts_retrieved']:>12}")
    print(f"  {'Successful':<25} {results_a['successful']:>12} {results_b['successful']:>12}")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Auto-benchmark RAG pipeline")
    parser.add_argument("--questions", type=int, default=5, help="Number of questions to generate")
    parser.add_argument("--input", type=str, help="Path to questions JSON (skip generation)")
    parser.add_argument("--output", type=str, default="eval/benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--model", type=str, help="Override LLM model")
    args = parser.parse_args()

    from config import config
    from src.retriever import HybridRAGRetriever
    from src.llm_factory import get_llm

    if args.model:
        config.llm_model = args.model

    print("=" * 50)
    print("  RAG Auto-Benchmark")
    print("=" * 50)

    retriever = HybridRAGRetriever(config)
    print(f"Docs in index: {retriever.doc_count}")

    if retriever.doc_count == 0:
        print("No documents found. Upload documents via app.py first.")
        sys.exit(1)

    llm = get_llm(config)

    # Load or generate questions
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions from {args.input}")
    else:
        print(f"\nGenerating {args.questions} evaluation questions...")
        questions = generate_questions_from_docs(retriever, llm, args.questions)
        if not questions:
            print("Failed to generate questions. Exiting.")
            sys.exit(1)

    # Run benchmark
    print(f"\nRunning benchmark on {len(questions)} questions...")
    results = run_benchmark(retriever, config, questions, args.output)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"  Questions    : {results['total_questions']}")
    print(f"  Successful   : {results['successful']}")
    print(f"  Avg Time     : {results['avg_time_seconds']}s")
    print(f"  Grade Pass   : {results['grade_pass_rate']:.1%}")
    print(f"  Avg Contexts : {results['avg_contexts_retrieved']}")
    print(f"{'='*50}")
