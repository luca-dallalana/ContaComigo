"""
Evaluation harness for the IRS Assistant RAG pipeline.

Runs the full pipeline (embed → retrieve → fuse → generate) against each
golden question in questions.yaml and scores on two dimensions:

  retrieval_hit  — the expected source document appeared in the top-K chunks
  answered       — the model gave a real answer (not the "não encontrei" fallback)

A question is considered passed only when both are true.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --output results.json
    python eval/run_eval.py --questions eval/questions_debug.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg2
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.client_factory import get_generation_client
from generation.ollama_client import OllamaConnectionError
from generation.prompt import build_prompt
from retrieval.bm25_search import bm25_search, build_bm25_index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.vector_search import SearchResult, embed_query, vector_search

load_dotenv()

POSTGRES_URL: str = os.environ["POSTGRES_URL"]
TOP_K: int = int(os.environ.get("TOP_K", 5))
VECTOR_WEIGHT: float = float(os.environ.get("VECTOR_WEIGHT", 0.7))
BM25_WEIGHT: float = float(os.environ.get("BM25_WEIGHT", 0.3))

QUESTIONS_FILE: Path = Path(__file__).parent / "questions.yaml"
QUESTION_DISPLAY_WIDTH: int = 50
FALLBACK_PHRASE: str = "não encontrei informação suficiente"


def _score(answer: str, chunks: list[SearchResult], case: dict) -> dict:
    retrieval_hit = any(case["expected_source"] in c.source_doc for c in chunks)
    answered = bool(answer) and FALLBACK_PHRASE not in answer.lower()
    passed = retrieval_hit and answered
    return {
        "question": case["question"],
        "retrieval_hit": retrieval_hit,
        "answered": answered,
        "passed": passed,
        "answer": answer,
        "sources": [c.source_doc for c in chunks],
    }


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Evaluate the IRS Assistant RAG pipeline.")
    arg_parser.add_argument("--output", type=str, help="Write full results to this JSON file.")
    arg_parser.add_argument("--questions", type=str, help="Path to a questions YAML file (default: eval/questions.yaml).")
    args = arg_parser.parse_args()

    questions_path = Path(args.questions) if args.questions else QUESTIONS_FILE
    cases = yaml.safe_load(questions_path.read_text(encoding="utf-8"))

    client = get_generation_client()
    if not client.health_check():
        print("ERROR: Ollama is not reachable or required models are missing.")
        sys.exit(1)

    conn = psycopg2.connect(POSTGRES_URL)
    bm25_index, chunk_ids = build_bm25_index(conn)

    results = []
    header = f"{'Question':<{QUESTION_DISPLAY_WIDTH}}  {'Ret':>3}  {'Ans':>3}  {'Pass':>4}"
    print(header)
    print("-" * len(header))

    for case in cases:
        question = case["question"]
        t0 = time.time()

        query_embedding = embed_query(question)
        vector_results = vector_search(query_embedding, conn, top_k=TOP_K * 2)
        bm25_results = bm25_search(question, bm25_index, chunk_ids, conn, top_k=TOP_K * 2)
        chunks = reciprocal_rank_fusion(
            vector_results, bm25_results, VECTOR_WEIGHT, BM25_WEIGHT
        )[:TOP_K]

        prompt = build_prompt(question, chunks)
        try:
            answer = "".join(client.generate(prompt, stream=False))
        except OllamaConnectionError as exc:
            print(f"  ERROR generating answer: {exc}")
            answer = ""

        result = _score(answer, chunks, case)
        result["elapsed_s"] = round(time.time() - t0, 1)
        results.append(result)

        q_display = question[:QUESTION_DISPLAY_WIDTH].ljust(QUESTION_DISPLAY_WIDTH)
        ret = "Y" if result["retrieval_hit"] else "N"
        ans = "Y" if result["answered"] else "N"
        passed = "Y" if result["passed"] else "N"
        print(f"{q_display}  {ret:>3}  {ans:>3}  {passed:>4}")

    conn.close()

    n = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print("-" * (QUESTION_DISPLAY_WIDTH + 16))
    print(f"Passed: {passed_count}/{n}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
