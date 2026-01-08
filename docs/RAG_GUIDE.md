# RAG (Retrieval-Augmented Generation) Guide

Complete guide to using RAG and Graph RAG in SciAgent.

## Table of Contents

- [Overview](#overview)
- [Traditional RAG](#traditional-rag)
- [Graph RAG](#graph-rag)
- [Hybrid RAG](#hybrid-rag)
- [Production Deployment](#production-deployment)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)

## Overview

SciAgent includes production-ready RAG capabilities:

- **Traditional RAG**: Vector-based semantic search with FAISS/ChromaDB
- **Graph RAG**: Knowledge graph-aware retrieval with relationship context
- **Hybrid RAG**: Combined approach for best results
- **RAG Agent**: Easy integration with SciAgent workflows

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   RAG System                         │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ Traditional  │  │  Graph RAG   │  │  Hybrid  │ │
│  │     RAG      │  │              │  │   RAG    │ │
│  │              │  │              │  │          │ │
│  │ • FAISS      │  │ • Knowledge  │  │ • Both   │ │
│  │ • ChromaDB   │  │   Graph      │  │ • Weight │ │
│  │ • Embeddings │  │ • Multi-hop  │  │ • Rerank │ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │          Embedding Model                      │  │
│  │   (sentence-transformers, OpenAI, etc.)      │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Traditional RAG

Traditional RAG uses vector embeddings for semantic search.

### Quick Start

```python
from sciagent.rag import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_type="faiss",  # or "chroma" or "memory"
    chunk_size=512,
    chunk_overlap=50,
)

# Add documents
documents = [
    "Dropout prevents overfitting in neural networks...",
    "Batch normalization improves training stability...",
]
rag.add_documents(documents)

# Query
results, context = rag.query("How to prevent overfitting?", top_k=5)

# Use context with LLM
prompt = f"Question: How to prevent overfitting?\n\nContext:\n{context}\n\nAnswer:"
```

### Vector Stores

#### FAISS (Recommended for Production)

```python
rag = RAGPipeline(
    vector_store_type="faiss",
    persist_directory=Path("rag_data/faiss"),
)
```

**Pros**: Fast, scalable, efficient
**Cons**: In-memory (but can save/load)
**Use case**: Production with <10M documents

#### ChromaDB (Recommended for Persistence)

```python
rag = RAGPipeline(
    vector_store_type="chroma",
    persist_directory=Path("rag_data/chroma"),
)
```

**Pros**: Persistent, SQL-backed, easy to use
**Cons**: Slower than FAISS
**Use case**: Development, small to medium scale

#### Memory (Testing Only)

```python
rag = RAGPipeline(vector_store_type="memory")
```

**Pros**: No dependencies
**Cons**: Not persistent, slow
**Use case**: Testing only

### Embedding Models

#### Sentence Transformers (Default)

```python
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384d
)
```

Options:
- `all-MiniLM-L6-v2`: Fast, 384d, good quality
- `all-mpnet-base-v2`: Slower, 768d, better quality
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A

#### OpenAI Embeddings

```python
rag = RAGPipeline(
    embedding_model="openai/text-embedding-ada-002"  # Requires API key
)
```

**Pros**: High quality
**Cons**: Costs money, requires API
**Use case**: Production with budget

### Chunking Strategies

```python
# Small chunks - better precision
rag = RAGPipeline(chunk_size=256, chunk_overlap=25)

# Large chunks - more context
rag = RAGPipeline(chunk_size=1024, chunk_overlap=100)

# Default - balanced
rag = RAGPipeline(chunk_size=512, chunk_overlap=50)
```

**Rules of thumb**:
- **Q&A**: Smaller chunks (256-512)
- **Summarization**: Larger chunks (512-1024)
- **Code**: Smaller chunks (256-384)
- **Papers**: Medium chunks (512-768)

### Re-ranking

```python
# Enable re-ranking for better quality
results, context = rag.query(
    query="Your question",
    top_k=5,
    rerank=True  # Slower but better
)
```

## Graph RAG

Graph RAG combines knowledge graphs with RAG for relationship-aware retrieval.

### Quick Start

```python
from sciagent.rag import GraphRAG
from sciagent.project.knowledge_graph import ScientificKnowledgeGraph

# Initialize knowledge graph
kg = ScientificKnowledgeGraph()

# Add papers (these become graph nodes)
from sciagent.utils.models import Paper

papers = [
    Paper(
        id="paper1",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        abstract="We propose the Transformer...",
        year=2017,
        citations=50000,
    ),
]

await kg.add_papers(papers)

# Initialize Graph RAG
graph_rag = GraphRAG(
    knowledge_graph=kg,
    max_hops=3,
)

# Query
results, context = graph_rag.query(
    "What papers discuss attention?",
    top_k=5
)
```

### Multi-Hop Retrieval

```python
# Traverse graph for complex queries
results = graph_rag.multi_hop_retrieve(
    query="How do transformers handle long sequences?",
    num_hops=2,
    top_k_per_hop=3,
)
```

**Use cases**:
- Finding indirect relationships
- Complex reasoning
- Cross-domain queries
- Causal chains

### Hybrid Search (Graph + Semantic)

```python
# Combine graph structure with semantic similarity
results, context = graph_rag.hybrid_search(
    query="Your question",
    top_k=5,
    alpha=0.7,  # 0.7 semantic + 0.3 structure
)
```

## Hybrid RAG

Combine traditional and graph RAG for best results.

```python
from sciagent.rag import HybridRAG

hybrid = HybridRAG(
    rag_pipeline=rag,
    graph_rag=graph_rag,
    combination_strategy="weighted",  # or "rerank"
)

# Query with weighting
results, context = hybrid.query(
    query="Your question",
    top_k=5,
    rag_weight=0.6,  # 60% RAG, 40% Graph RAG
)
```

**When to use**:
- Best quality (combines both)
- Complex queries
- When you have both documents and graph
- Production systems

## RAG Agent

Easy integration with SciAgent.

```python
from sciagent.agents import RAGAgent
from sciagent.utils.config import Config

# Initialize
config = Config()
rag_agent = RAGAgent(config)

# Add documents
await rag_agent.add_documents([
    "Document 1 content...",
    "Document 2 content...",
])

# Query
result = await rag_agent.query(
    query="Your question",
    top_k=5,
    mode="hybrid",  # "rag", "graph", or "hybrid"
)

print(result["context"])
```

### Loading Documents from Directory

```python
rag_agent.load_corpus_from_directory(
    directory=Path("docs/"),
    file_pattern="*.txt"
)
```

### Persistence

```python
# Save
rag_agent.save(Path("rag_agent_data"))

# Load
rag_agent.load(Path("rag_agent_data"))
```

## Production Deployment

### Architecture

```
┌──────────────────────────────────────────┐
│         Load Balancer                     │
└──────────────┬───────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌────▼────┐
│ RAG    │          │  RAG    │
│ Server │          │ Server  │
│   1    │          │    2    │
└───┬────┘          └────┬────┘
    │                    │
    └─────────┬──────────┘
              │
       ┌──────▼──────┐
       │   FAISS     │
       │   Cluster   │
       └─────────────┘
```

### Configuration

```python
# config.yaml
rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_store: "faiss"
  chunk_size: 512
  chunk_overlap: 50
  batch_size: 32  # For embedding
  max_concurrent: 10  # Concurrent queries
  cache_ttl: 3600  # Cache results for 1 hour

faiss:
  index_type: "IVF"  # Inverted file index
  nlist: 100  # Number of clusters
  nprobe: 10  # Number of clusters to search

persistence:
  enabled: true
  directory: "/data/rag"
  backup_interval: 3600  # 1 hour
```

### Scaling

#### Horizontal Scaling

```python
# Deploy multiple RAG servers
# Each server loads the same index
# Use load balancer to distribute queries

from sciagent.rag import RAGPipeline

# On each server
rag = RAGPipeline(vector_store_type="faiss")
rag.load(Path("/shared/rag_index"))

# Serve queries
def query_endpoint(query: str):
    results, context = rag.query(query)
    return {"results": results, "context": context}
```

#### Vertical Scaling

```python
# Optimize FAISS index
import faiss

# Use GPU
rag = RAGPipeline(
    vector_store_type="faiss",
    device="cuda",  # Use GPU
)

# Optimize index
# After loading, optimize for your hardware
index = rag.vector_store.index
index_optimized = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(), 0, index
)
```

### Monitoring

```python
import time
from prometheus_client import Counter, Histogram

# Metrics
queries_total = Counter('rag_queries_total', 'Total queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')

def monitored_query(query: str):
    queries_total.inc()

    start = time.time()
    results, context = rag.query(query)
    duration = time.time() - start

    query_duration.observe(duration)

    return results, context
```

## Best Practices

### 1. Document Preparation

```python
# Clean documents before adding
def clean_document(text: str) -> str:
    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove special characters if needed
    # text = re.sub(r'[^\w\s]', '', text)

    # Truncate very long documents
    if len(text) > 100000:
        text = text[:100000]

    return text

documents = [clean_document(doc) for doc in raw_documents]
rag.add_documents(documents)
```

### 2. Query Optimization

```python
# Expand queries for better retrieval
def expand_query(query: str) -> str:
    # Add synonyms, related terms
    expansions = {
        "ML": "machine learning",
        "NN": "neural network",
        "DL": "deep learning",
    }

    for abbrev, full in expansions.items():
        query = query.replace(abbrev, full)

    return query

query = expand_query("How does ML work?")
results, context = rag.query(query)
```

### 3. Context Management

```python
# Limit context size for LLM
MAX_CONTEXT_LENGTH = 4000

def truncate_context(context: str, max_length: int) -> str:
    if len(context) <= max_length:
        return context

    # Truncate to sentences
    sentences = context.split(". ")
    truncated = ""

    for sentence in sentences:
        if len(truncated) + len(sentence) < max_length:
            truncated += sentence + ". "
        else:
            break

    return truncated

context = truncate_context(context, MAX_CONTEXT_LENGTH)
```

### 4. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(query: str, top_k: int = 5):
    return rag.query(query, top_k=top_k)

# Subsequent identical queries are cached
result1 = cached_query("What is dropout?")
result2 = cached_query("What is dropout?")  # Instant
```

### 5. Batch Processing

```python
# Process queries in batches
queries = ["Query 1", "Query 2", "Query 3", ...]

# Batch embed queries
query_embeddings = rag.embedding_model.embed(queries)

# Batch search
results_batch = [
    rag.vector_store.search(emb, top_k=5)
    for emb in query_embeddings
]
```

## Advanced Topics

### Custom Embedding Models

```python
class CustomEmbedding:
    def __init__(self):
        # Load your model
        self.model = load_model()

    def embed(self, texts):
        return self.model.encode(texts)

# Use custom model
rag.embedding_model.model = CustomEmbedding()
```

### Custom Rerankers

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_with_cross_encoder(query, results):
    pairs = [[query, r.document.content] for r in results]
    scores = reranker.predict(pairs)

    for result, score in zip(results, scores):
        result.score = score

    results.sort(key=lambda r: r.score, reverse=True)
    return results
```

### Hybrid Search with BM25

```python
from rank_bm25 import BM25Okapi

# Combine dense (embeddings) with sparse (BM25)
class HybridRetriever:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline

        # Build BM25 index
        documents = [d.content for d in rag_pipeline.vector_store.documents]
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5, alpha=0.7):
        # Dense search
        dense_results = self.rag.retrieve(query, top_k=top_k*2)

        # Sparse search
        sparse_scores = self.bm25.get_scores(query.split())

        # Combine
        combined = []
        for result in dense_results:
            idx = result.rank
            combined_score = (
                alpha * result.score +
                (1 - alpha) * sparse_scores[idx]
            )
            result.score = combined_score
            combined.append(result)

        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[:top_k]
```

### Fine-tuning Embeddings

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data
train_examples = [
    InputExample(texts=['query', 'positive_doc']),
    # ... more examples
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Train
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
)

# Use fine-tuned model
rag = RAGPipeline(embedding_model=model)
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
rag.embedding_model.batch_size = 8

# Use smaller model
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # 384d instead of 768d
)

# Use CPU
rag = RAGPipeline(device="cpu")
```

### Slow Queries

```python
# Use FAISS instead of memory
rag = RAGPipeline(vector_store_type="faiss")

# Reduce top_k
results = rag.query(query, top_k=3)  # instead of 10

# Disable reranking for speed
results = rag.retrieve(query, top_k=5, rerank=False)
```

### Poor Quality Results

```python
# Increase chunk overlap
rag = RAGPipeline(chunk_size=512, chunk_overlap=100)

# Use better embedding model
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# Enable reranking
results = rag.retrieve(query, rerank=True)

# Use hybrid search
results = hybrid_rag.query(query)
```

## Resources

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ChromaDB](https://www.trychroma.com/)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex](https://docs.llamaindex.ai/)

## Next Steps

1. **Try the examples**: Run `examples/rag_examples.py`
2. **Load your data**: Add your documents to RAG
3. **Tune parameters**: Experiment with chunk size, models
4. **Monitor performance**: Track query times and quality
5. **Scale up**: Deploy to production

---

For more help, see:
- [SciAgent Documentation](../README.md)
- [Examples](../examples/)
- [GitHub Issues](https://github.com/yourusername/sciagent/issues)
