# tiny-rag

A lightweight RAG (Retrieval-Augmented Generation) package that runs entirely on CPU without requiring GPUs or cloud services.

## Overview

tiny-rag is designed to make RAG accessible to everyone by removing the typical hardware and service dependencies. While traditional RAG systems require GPU-accelerated embedding models and reranker models (like OpenAI's text-embedding services), tiny-rag provides a fully functional RAG implementation that runs efficiently on CPU-only environments.

tiny-rag leverages [static-embedding-japanese](https://huggingface.co/hotchpotch/static-embedding-japanese), an ultra-fast embedding model that achieves 126x faster CPU inference compared to traditional transformer models while maintaining strong performance on Japanese text tasks.

## Key Features

- **CPU-Only Execution**: No GPU required - runs on any standard computer
- **No Cloud Dependencies**: Fully offline operation without external API calls
- **Lightweight**: Minimal resource footprint for embedding and reranking
- **Easy to Use**: Simple API for quick integration into your projects
- **Cost-Effective**: No cloud service fees or expensive hardware requirements
- **Japanese Language Support**: Currently optimized for Japanese text processing

## Language Support

⚠️ **Important**: tiny-rag currently supports **Japanese language only**. Support for additional languages is planned for future releases.

## Installation

```bash
pip install tiny-rag
```

## Quick Start

```python
from tiny_rag import TinyRAG

# Initialize tiny-rag
rag = TinyRAG()

# Add documents (Japanese text)
rag.add_documents([
    "ドキュメント1の内容...",
    "ドキュメント2の内容...",
])

# Query (in Japanese)
results = rag.query("あなたの質問をここに入力")
```

## Use Cases

- **Local Development**: Test RAG pipelines without cloud costs
- **Edge Computing**: Deploy RAG on resource-constrained devices
- **Privacy-Sensitive Applications**: Keep all data processing local
- **Educational Projects**: Learn RAG concepts without infrastructure overhead
- **Japanese Text Processing**: Optimized for Japanese language applications

## Technical Details

### Embedding Model

tiny-rag uses [static-embedding-japanese](https://huggingface.co/hotchpotch/static-embedding-japanese), which provides:

- **Ultra-fast CPU Performance**: 126x faster inference than comparable transformer models
- **1024-dimensional embeddings**: Can be reduced to 32-512 dimensions for efficiency
- **Simple Architecture**: Uses token embedding averaging instead of complex attention mechanisms
- **Strong Benchmark Performance**: JMTEB score of 67.17 (micro-average)
- **Matryoshka Representation Learning**: Enables efficient dimension reduction without retraining

### Reranker Model

For improved retrieval accuracy, tiny-rag employs [japanese-reranker-xsmall-v2](https://huggingface.co/hotchpotch/japanese-reranker-xsmall-v2):

- **Compact Size**: Only 36.8M parameters with 10 layers
- **High Performance**: Average score of 0.8699 on Japanese benchmarks
- **CPU-Friendly**: Designed to run at practical speeds on CPU and Apple Silicon
- **Modern Architecture**: Based on ModernBert for efficient text ranking
- **Excellent Benchmark Results**: JaCWIR (0.9409), JSQuAD (0.9776), MIRACL (0.8206)

Both models are specifically chosen for their exceptional CPU performance while maintaining high-quality results for Japanese text processing.

## Benchmark Performance

tiny-rag has been evaluated on standard Japanese retrieval benchmarks to demonstrate its effectiveness:

### Datasets

- **JQaRA** (Japanese Question Answering with Retrieval Augmentation): 144,372 documents
- **JaCWIR** (Japanese Casual Web IR): 513,107 documents

### Results

| Dataset | NDCG@10 | MRR@10 | MAP@10 | Hits@10 | Avg Query Time |
|---------|---------|---------|---------|---------|----------------|
| JQaRA   | 0.7944  | 0.8000  | -       | -       | 0.755 sec      |
| JaCWIR  | -       | -       | 0.8000  | 0.8000  | 1.185 sec      |

### Running Benchmarks

```bash
# Quick test with 5 queries per dataset
make bench

# Full benchmark evaluation
make bench-full

# Custom benchmark
python -m bench.benchmark --dimensions 512 --max-queries 10
```

### Benchmark Options

- `--dimensions`: Embedding dimensions (32, 64, 128, 256, 512, 1024) - Default: 1024
- `--max-queries`: Maximum queries per dataset (for testing)

### Performance Insights

- **High Accuracy**: Achieves 0.79-0.80 scores across all metrics
- **Practical Speed**: Query processing under 1.2 seconds even for large datasets
- **Scalable**: Performance scales reasonably with dataset size
- **CPU-Friendly**: All processing runs efficiently on standard hardware

## Requirements

- Python 3.13+
- No GPU required
- Minimal RAM requirements

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/sonesuke/tiny-rag.git
cd tiny-rag

# Install development dependencies
uv sync

# Run tests
uv run pytest --cov=src

# Run benchmarks
make bench       # Quick test (5 queries)
make bench-full  # Full evaluation
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **sonesuke** - [GitHub](https://github.com/sonesuke)

## Acknowledgments

Built with a focus on accessibility and efficiency, making RAG technology available to everyone regardless of hardware limitations.