"""TinyRAG benchmark using JQaRA and JaCWIR datasets."""

import argparse
import time

from tiny_rag import TinyRAG

from .metrics import aggregate_metrics, evaluate_retrieval_metrics
from .utils import (
    create_progress_bar,
    get_relevant_docs,
    load_jacwir_dataset,
    load_jqara_dataset,
    prepare_corpus_documents,
    prepare_queries,
    print_results_table,
)


def evaluate_on_jqara(rag: TinyRAG, max_queries: int | None = None) -> dict[str, float]:
    """Evaluate TinyRAG on JQaRA dataset.

    Args:
        rag: TinyRAG instance.
        max_queries: Maximum number of queries to evaluate (for testing).

    Returns:
        Dictionary containing averaged metrics.
    """
    print("Loading JQaRA dataset...")
    load_start = time.time()
    queries, corpus = load_jqara_dataset()
    load_time = time.time() - load_start

    # Prepare data
    doc_map = prepare_corpus_documents(corpus)
    prepared_queries = prepare_queries(queries)

    if max_queries:
        prepared_queries = prepared_queries[:max_queries]

    # Add documents to RAG
    print(f"Adding {len(doc_map)} documents to TinyRAG...")
    add_start = time.time()
    documents = list(doc_map.values())
    rag.add_documents(documents)
    add_time = time.time() - add_start
    print(f"Document loading time: {add_time:.2f} seconds")

    # Create mapping from document text back to doc_id
    text_to_doc_id = {text: doc_id for doc_id, text in doc_map.items()}

    # Evaluate queries
    all_metrics = []
    query_start = time.time()
    pbar = create_progress_bar(len(prepared_queries), "JQaRA evaluation")

    for query_item in prepared_queries:
        query_text = query_item["query"]
        relevant_docs = get_relevant_docs(query_item)

        # Get results from TinyRAG
        results = rag.query(query_text, top_k=10)

        # Convert results to (doc_id, score) format
        ranked_results = []
        for result in results:
            doc_text = result.document
            if doc_text in text_to_doc_id:
                doc_id = text_to_doc_id[doc_text]
                ranked_results.append((doc_id, result.similarity))

        # Calculate metrics
        metrics = evaluate_retrieval_metrics(ranked_results, relevant_docs, k=10)
        all_metrics.append(metrics)

        pbar.update(1)

    pbar.close()
    query_time = time.time() - query_start
    avg_query_time = query_time / len(prepared_queries)
    print(f"Query time: {query_time:.2f} seconds ({avg_query_time:.3f} sec/query)")

    # Return averaged metrics (focus on NDCG@10 and MRR@10 for JQaRA)
    aggregated = aggregate_metrics(all_metrics)
    return {
        "ndcg@10": aggregated.get("ndcg@10", 0.0),
        "mrr@10": aggregated.get("mrr@10", 0.0),
        "load_time_sec": load_time,
        "add_documents_time_sec": add_time,
        "query_time_sec": query_time,
        "avg_query_time_sec": avg_query_time,
    }


def evaluate_on_jacwir(rag: TinyRAG, max_queries: int | None = None) -> dict[str, float]:
    """Evaluate TinyRAG on JaCWIR dataset.

    Args:
        rag: TinyRAG instance.
        max_queries: Maximum number of queries to evaluate (for testing).

    Returns:
        Dictionary containing averaged metrics.
    """
    print("Loading JaCWIR dataset...")
    load_start = time.time()
    queries, corpus = load_jacwir_dataset()
    load_time = time.time() - load_start

    # Prepare data
    doc_map = prepare_corpus_documents(corpus)
    prepared_queries = prepare_queries(queries)

    if max_queries:
        prepared_queries = prepared_queries[:max_queries]

    # Add documents to RAG
    print(f"Adding {len(doc_map)} documents to TinyRAG...")
    add_start = time.time()
    documents = list(doc_map.values())
    rag.add_documents(documents)
    add_time = time.time() - add_start
    print(f"Document loading time: {add_time:.2f} seconds")

    # Create mapping from document text back to doc_id
    text_to_doc_id = {text: doc_id for doc_id, text in doc_map.items()}

    # Evaluate queries
    all_metrics = []
    query_start = time.time()
    pbar = create_progress_bar(len(prepared_queries), "JaCWIR evaluation")

    for query_item in prepared_queries:
        query_text = query_item["query"]
        relevant_docs = get_relevant_docs(query_item)

        # Get results from TinyRAG
        results = rag.query(query_text, top_k=10)

        # Convert results to (doc_id, score) format
        ranked_results = []
        for result in results:
            doc_text = result.document
            if doc_text in text_to_doc_id:
                doc_id = text_to_doc_id[doc_text]
                ranked_results.append((doc_id, result.similarity))

        # Calculate metrics
        metrics = evaluate_retrieval_metrics(ranked_results, relevant_docs, k=10)
        all_metrics.append(metrics)

        pbar.update(1)

    pbar.close()
    query_time = time.time() - query_start
    avg_query_time = query_time / len(prepared_queries)
    print(f"Query time: {query_time:.2f} seconds ({avg_query_time:.3f} sec/query)")

    # Return averaged metrics (focus on MAP@10 and Hits@10 for JaCWIR)
    aggregated = aggregate_metrics(all_metrics)
    return {
        "map@10": aggregated.get("map@10", 0.0),
        "hits@10": aggregated.get("hits@10", 0.0),
        "load_time_sec": load_time,
        "add_documents_time_sec": add_time,
        "query_time_sec": query_time,
        "avg_query_time_sec": avg_query_time,
    }


def run_benchmark(dimensions: int = 1024, max_queries: int | None = None) -> dict[str, dict[str, float]]:
    """Run full benchmark on both datasets.

    Args:
        dimensions: Embedding dimensions for TinyRAG.
        max_queries: Maximum queries per dataset (for testing).

    Returns:
        Dictionary with results for each dataset.
    """
    print(f"Starting TinyRAG benchmark with {dimensions} dimensions")
    start_time = time.time()

    results = {}

    # JQaRA evaluation
    print("\n=== JQaRA Evaluation ===")
    rag_jqara = TinyRAG(dimensions=dimensions)
    jqara_results = evaluate_on_jqara(rag_jqara, max_queries)
    results["JQaRA"] = jqara_results

    # JaCWIR evaluation
    print("\n=== JaCWIR Evaluation ===")
    rag_jacwir = TinyRAG(dimensions=dimensions)
    jacwir_results = evaluate_on_jacwir(rag_jacwir, max_queries)
    results["JaCWIR"] = jacwir_results

    # Print results
    print_results_table(results)

    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.2f} seconds")

    return results


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Run TinyRAG benchmark on JQaRA and JaCWIR datasets")
    parser.add_argument(
        "--dimensions", type=int, default=1024, choices=[32, 64, 128, 256, 512, 1024], help="Embedding dimensions"
    )
    parser.add_argument("--max-queries", type=int, help="Maximum queries per dataset (for testing)")

    args = parser.parse_args()

    run_benchmark(dimensions=args.dimensions, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
