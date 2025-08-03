"""Utilities for JMTEB benchmark evaluation."""

from collections.abc import Iterator
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_jqara_dataset(split: str = "test") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load JQaRA dataset directly.

    Args:
        split: Dataset split to load.

    Returns:
        Tuple of (queries, corpus) lists.
    """
    # Load JQaRA dataset directly (it's in reranking format)
    dataset = load_dataset("hotchpotch/JQaRA", split=split)

    # Group by query and collect passages
    queries_dict = {}
    corpus = {}

    for item in dataset:
        q_id = item["q_id"]
        question = item["question"]
        passage_text = item["text"]
        passage_id = item["passage_row_id"]
        label = item["label"]  # 1 for positive, 0 for negative

        # Add to corpus
        corpus[passage_id] = passage_text

        # Group queries
        if q_id not in queries_dict:
            queries_dict[q_id] = {"query_id": q_id, "query": question, "positive_passages": [], "negative_passages": []}

        # Add passage reference based on label
        passage_ref = {"doc_id": passage_id, "text": passage_text}
        if label == 1:
            queries_dict[q_id]["positive_passages"].append(passage_ref)
        else:
            queries_dict[q_id]["negative_passages"].append(passage_ref)

    # Convert to lists
    queries = list(queries_dict.values())
    corpus_list = [{"doc_id": doc_id, "text": text} for doc_id, text in corpus.items()]

    return queries, corpus_list


def load_jacwir_dataset(split: str = "eval") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load JaCWIR dataset directly.

    Args:
        split: Dataset split to load.

    Returns:
        Tuple of (queries, corpus) lists.
    """
    # Load JaCWIR eval dataset and collection
    eval_dataset = load_dataset("hotchpotch/JaCWIR", "eval", split=split)
    collection_dataset = load_dataset("hotchpotch/JaCWIR", "collection", split="collection")

    # Build doc_id to text mapping from collection
    doc_to_text = {}
    for item in collection_dataset:
        doc_id = item["doc_id"]
        # Combine title and description as text
        title = item.get("title", "")
        description = item.get("description", "")
        text = f"{title}\n{description}".strip()
        doc_to_text[doc_id] = text

    # Process eval dataset
    queries = []
    for i, item in enumerate(eval_dataset):
        query_text = item["query"]
        positive_doc_id = item["positive"]
        negative_doc_ids = item["negatives"]

        # Create positive passages
        positive_passages = []
        if positive_doc_id in doc_to_text:
            positive_passages.append({"doc_id": positive_doc_id, "text": doc_to_text[positive_doc_id]})

        # Create negative passages
        negative_passages = []
        for neg_doc_id in negative_doc_ids:
            if neg_doc_id in doc_to_text:
                negative_passages.append({"doc_id": neg_doc_id, "text": doc_to_text[neg_doc_id]})

        query_item = {
            "query_id": str(i),
            "query": query_text,
            "positive_passages": positive_passages,
            "negative_passages": negative_passages,
        }
        queries.append(query_item)

    # Create corpus from collection
    corpus_list = [{"doc_id": doc_id, "text": text} for doc_id, text in doc_to_text.items()]

    return queries, corpus_list


def prepare_corpus_documents(corpus: list[dict[str, Any]]) -> dict[str, str]:
    """Prepare corpus documents for TinyRAG.

    Args:
        corpus: List of corpus items with 'doc_id' and 'text' fields.

    Returns:
        Dictionary mapping doc_id to text.
    """
    doc_map = {}
    for item in corpus:
        doc_id = item["doc_id"]
        text = item["text"]
        doc_map[doc_id] = text

    return doc_map


def prepare_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare queries for evaluation.

    Args:
        queries: List of query items.

    Returns:
        List of prepared query dictionaries.
    """
    prepared_queries = []

    for i, query in enumerate(queries):
        # Use index as query_id if not present
        query_id = query.get("query_id", query.get("id", str(i)))

        prepared_query = {
            "query_id": query_id,
            "query": query["query"],
            "positive_passages": query.get("positive_passages", []),
            "negative_passages": query.get("negative_passages", []),
        }
        prepared_queries.append(prepared_query)

    return prepared_queries


def get_relevant_docs(query_item: dict[str, Any]) -> set[str]:
    """Extract relevant document IDs for a query.

    Args:
        query_item: Query item with positive_passages.

    Returns:
        Set of relevant document IDs.
    """
    relevant_docs = set()

    if "positive_passages" in query_item:
        for passage in query_item["positive_passages"]:
            if isinstance(passage, dict) and "doc_id" in passage:
                relevant_docs.add(passage["doc_id"])
            elif isinstance(passage, str):
                # Sometimes doc_id is directly stored as string
                relevant_docs.add(passage)

    return relevant_docs


def save_results(results: dict[str, Any], output_file: str) -> None:
    """Save benchmark results to file.

    Args:
        results: Dictionary containing benchmark results.
        output_file: Path to output file.
    """
    df = pd.DataFrame([results])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print results in a formatted table.

    Args:
        results: Dictionary with dataset names as keys and metrics as values.
    """
    print("\n=== TinyRAG Benchmark Results ===")

    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        print("-" * 50)

        # Print performance metrics first
        for metric_name, score in metrics.items():
            if not metric_name.endswith("_time_sec"):
                print(f"{metric_name:<20}: {score:.4f}")

        # Print timing metrics
        print("\nTiming:")
        time_metrics = {k: v for k, v in metrics.items() if k.endswith("_time_sec")}
        for metric_name, time_val in time_metrics.items():
            display_name = metric_name.replace("_time_sec", "").replace("_", " ").title()
            if "avg" in metric_name.lower():
                print(f"{display_name:<20}: {time_val:.3f} sec")
            else:
                print(f"{display_name:<20}: {time_val:.2f} sec")


def batch_queries(queries: list[dict[str, Any]], batch_size: int = 100) -> Iterator[list[dict[str, Any]]]:
    """Batch queries for processing.

    Args:
        queries: List of queries.
        batch_size: Size of each batch.

    Yields:
        Batches of queries.
    """
    for i in range(0, len(queries), batch_size):
        yield queries[i : i + batch_size]


def create_progress_bar(total: int, desc: str = "Processing") -> tqdm:
    """Create a progress bar.

    Args:
        total: Total number of items.
        desc: Description for the progress bar.

    Returns:
        tqdm progress bar instance.
    """
    return tqdm(total=total, desc=desc, unit="query")
