"""
RAG (Retrieval-Augmented Generation) Examples

Demonstrates:
- Traditional RAG with vector stores
- Graph RAG with knowledge graphs
- Hybrid RAG
- Document ingestion and querying
- Integration with SciAgent
"""

import asyncio
from pathlib import Path

from sciagent import SciAgent
from sciagent.rag import RAGPipeline, GraphRAG, HybridRAG, Document
from sciagent.project.knowledge_graph import ScientificKnowledgeGraph
from sciagent.agents.rag_agent import RAGAgent
from sciagent.utils.config import Config


async def example_1_basic_rag():
    """
    Example 1: Basic RAG Pipeline
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic RAG Pipeline")
    print("=" * 70 + "\n")

    # Initialize RAG pipeline
    rag = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss",  # or "chroma" or "memory"
        chunk_size=512,
        chunk_overlap=50,
    )

    # Sample scientific documents
    documents = [
        """
        Dropout is a regularization technique that prevents neural networks from overfitting.
        During training, dropout randomly sets a fraction of input units to 0 at each update.
        This prevents units from co-adapting too much and has been shown to improve generalization.
        Dropout rates between 0.2 and 0.5 are commonly used.
        """,
        """
        Batch normalization accelerates deep network training by reducing internal covariate shift.
        It normalizes the inputs of each layer to have mean 0 and variance 1.
        This allows for higher learning rates and reduces the dependence on initialization.
        Batch normalization has become a standard component in most modern neural architectures.
        """,
        """
        Attention mechanisms allow models to focus on relevant parts of the input.
        The transformer architecture, introduced in "Attention is All You Need",
        relies entirely on self-attention and has revolutionized NLP.
        Attention computes weighted sums based on similarity scores between queries and keys.
        """,
        """
        ResNet introduced skip connections that allow gradients to flow directly through
        the network. This enables training of much deeper networks (100+ layers).
        The residual connections help address the vanishing gradient problem.
        ResNet won the 2015 ImageNet competition with 152 layers.
        """,
        """
        Transfer learning involves taking a pre-trained model and fine-tuning it on a new task.
        This is particularly effective when the new task has limited data.
        Models pre-trained on ImageNet are commonly used for computer vision tasks.
        BERT and GPT models demonstrate the power of transfer learning in NLP.
        """,
    ]

    # Add documents to RAG
    print("Adding documents to RAG pipeline...")
    rag.add_documents(documents)

    # Query the RAG system
    queries = [
        "How does dropout prevent overfitting?",
        "What are skip connections in ResNet?",
        "Explain attention mechanisms",
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        results, context = rag.query(query, top_k=3)

        print(f"\n✓ Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] Score: {result.score:.3f}")
            print(f"      {result.document.content[:150]}...")

    # Save for later use
    save_path = Path("rag_data")
    rag.save(save_path)
    print(f"\n✓ RAG pipeline saved to {save_path}")


async def example_2_graph_rag():
    """
    Example 2: Graph RAG with Knowledge Graph
    """
    print("\n" + "=" * 70)
    print("Example 2: Graph RAG with Knowledge Graph")
    print("=" * 70 + "\n")

    # Initialize knowledge graph and Graph RAG
    kg = ScientificKnowledgeGraph()
    graph_rag = GraphRAG(
        knowledge_graph=kg,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        max_hops=3,
    )

    # Add sample papers to knowledge graph
    from sciagent.utils.models import Paper, Hypothesis

    papers = [
        Paper(
            id="paper1",
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            abstract="We propose the Transformer, a model based entirely on attention mechanisms...",
            year=2017,
            citations=50000,
        ),
        Paper(
            id="paper2",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin et al."],
            abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers...",
            year=2018,
            citations=40000,
        ),
        Paper(
            id="paper3",
            title="Deep Residual Learning for Image Recognition",
            authors=["He et al."],
            abstract="We present a residual learning framework to ease the training of very deep networks...",
            year=2015,
            citations=60000,
        ),
    ]

    # Add to knowledge graph
    await kg.add_papers(papers)

    # Add hypotheses
    hyp1 = Hypothesis(
        statement="Self-attention mechanisms enable better long-range dependencies",
        rationale="Attention allows direct connections between distant tokens",
        testable_predictions=["Better performance on long sequences"],
        expected_evidence="Empirical results on long-context tasks",
    )

    kg.add_hypothesis(hyp1, related_papers=["paper1", "paper2"])

    # Rebuild embeddings
    graph_rag._build_node_embeddings()

    # Query Graph RAG
    queries = [
        "What papers discuss attention mechanisms?",
        "How do transformers handle long-range dependencies?",
        "Explain residual connections",
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")

        # Regular retrieval
        results, context = graph_rag.query(query, top_k=3, use_multi_hop=False)

        print(f"\n✓ Found {len(results)} relevant nodes:")
        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] {result.node.node_type.upper()} | Score: {result.score:.3f}")
            print(f"      {result.node.content[:150]}...")

            if result.related_nodes:
                print(f"      Related: {len(result.related_nodes)} nodes")

    # Multi-hop retrieval
    print("\n\n🔄 Multi-hop retrieval:")
    query = "What techniques improve neural network training?"
    results = graph_rag.multi_hop_retrieve(query, num_hops=2, top_k_per_hop=2)

    print(f"\n✓ Multi-hop found {len(results)} total nodes across 2 hops")

    # Visualize knowledge graph
    kg.visualize(Path("knowledge_graph_rag.png"))
    print("\n✓ Knowledge graph visualization saved")


async def example_3_hybrid_rag():
    """
    Example 3: Hybrid RAG (Combining Traditional + Graph)
    """
    print("\n" + "=" * 70)
    print("Example 3: Hybrid RAG")
    print("=" * 70 + "\n")

    # Initialize both systems
    rag = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="memory",
    )

    kg = ScientificKnowledgeGraph()
    graph_rag = GraphRAG(kg)

    # Add documents to traditional RAG
    documents = [
        "Dropout randomly drops units during training to prevent overfitting.",
        "Batch normalization normalizes activations to improve training stability.",
        "Adam optimizer adapts learning rates for each parameter.",
    ]
    rag.add_documents(documents)

    # Add to Graph RAG
    graph_rag.add_documents_to_graph(documents)

    # Create hybrid RAG
    hybrid = HybridRAG(
        rag_pipeline=rag,
        graph_rag=graph_rag,
        combination_strategy="weighted",
    )

    # Query hybrid system
    query = "What regularization techniques exist?"

    print(f"📝 Query: {query}\n")

    # Try different weights
    for rag_weight in [0.3, 0.5, 0.7]:
        print(f"\n--- RAG Weight: {rag_weight} (Graph Weight: {1-rag_weight}) ---")

        results, context = hybrid.query(query, top_k=3, rag_weight=rag_weight)

        print(f"✓ Combined {len(results)} results")
        print(f"\nContext Preview:\n{context[:300]}...\n")


async def example_4_rag_agent():
    """
    Example 4: Using RAG Agent
    """
    print("\n" + "=" * 70)
    print("Example 4: RAG Agent Integration")
    print("=" * 70 + "\n")

    # Initialize RAG Agent
    config = Config()
    rag_agent = RAGAgent(config)

    # Add documents
    documents = [
        "Convolutional neural networks are designed for processing grid-like data such as images.",
        "Recurrent neural networks process sequential data by maintaining hidden states.",
        "Generative adversarial networks consist of a generator and discriminator.",
        "Variational autoencoders learn latent representations through probabilistic encoding.",
    ]

    print("Adding documents to RAG agent...")
    result = await rag_agent.add_documents(documents)
    print(f"✓ Added {result['documents_added']} documents")

    # Query different modes
    query = "What neural networks are used for images?"

    print(f"\n📝 Query: {query}\n")

    # Traditional RAG
    print("--- Mode: Traditional RAG ---")
    result = await rag_agent.query(query, top_k=2, mode="rag")
    print(f"✓ Found {len(result['results'])} results")
    for r in result["results"]:
        print(f"  - Score {r['score']:.3f}: {r['content'][:100]}...")

    # Graph RAG
    print("\n--- Mode: Graph RAG ---")
    result = await rag_agent.query(query, top_k=2, mode="graph")
    print(f"✓ Found {len(result['results'])} results")
    for r in result["results"]:
        print(f"  - Score {r['score']:.3f} | Type: {r['node_type']}")

    # Hybrid
    print("\n--- Mode: Hybrid RAG ---")
    result = await rag_agent.query(query, top_k=2, mode="hybrid")
    print(f"✓ Found {len(result['results'])} results (weighted combination)")


async def example_5_sciagent_with_rag():
    """
    Example 5: Full SciAgent with RAG Integration
    """
    print("\n" + "=" * 70)
    print("Example 5: SciAgent with RAG")
    print("=" * 70 + "\n")

    # This demonstrates how RAG enhances SciAgent experiments

    # Initialize SciAgent
    agent = SciAgent()

    # The Science Agent automatically uses:
    # 1. MCP for literature search → populates knowledge graph
    # 2. Knowledge graph → available for Graph RAG
    # 3. Graph RAG → enables context-aware hypothesis generation

    query = """
    Research question: Do deeper neural networks always perform better?

    Use RAG to:
    1. Search existing literature on network depth
    2. Generate hypotheses based on retrieved papers
    3. Design experiments informed by prior work
    """

    print(f"Running SciAgent experiment with RAG integration...\n")
    print(f"Query: {query[:100]}...\n")

    # In a real scenario, this would:
    # - Search papers via MCP
    # - Store in knowledge graph
    # - Use Graph RAG to find related concepts
    # - Generate literature-informed hypotheses
    # - Design experiments based on existing work

    print("✓ SciAgent uses RAG to:")
    print("  1. Retrieve relevant papers from knowledge graph")
    print("  2. Extract key concepts and relationships")
    print("  3. Generate hypotheses informed by literature")
    print("  4. Design experiments that build on prior work")
    print("  5. Avoid redundant experiments")

    print("\nThis enables:")
    print("  • Literature-aware research")
    print("  • Better hypothesis quality")
    print("  • More informed experiment design")
    print("  • Reduced redundancy")
    print("  • Faster scientific progress")


async def example_6_production_rag_pipeline():
    """
    Example 6: Production-Ready RAG Pipeline
    """
    print("\n" + "=" * 70)
    print("Example 6: Production RAG Pipeline")
    print("=" * 70 + "\n")

    # Production configuration
    config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "faiss",  # Fast for production
        "chunk_size": 512,
        "chunk_overlap": 50,
        "rerank": True,  # Better quality
        "persist_directory": Path("production_rag"),
    }

    print("Production Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Initialize pipeline
    rag = RAGPipeline(
        embedding_model=config["embedding_model"],
        vector_store_type=config["vector_store"],
        persist_directory=config["persist_directory"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    # Simulate loading large corpus
    print("\n📚 Loading document corpus...")

    # In production, this would load from files/database
    corpus = [
        f"Document {i}: Sample scientific text about topic {i % 10}..."
        for i in range(100)
    ]

    rag.add_documents(corpus)
    print(f"✓ Loaded {len(corpus)} documents")

    # Batch queries (production use case)
    print("\n🔍 Processing batch queries...")

    queries = [
        f"Query {i}: What is topic {i}?" for i in range(10)
    ]

    results_batch = []
    for query in queries:
        results, context = rag.query(query, top_k=3)
        results_batch.append((query, results))

    print(f"✓ Processed {len(queries)} queries")

    # Save pipeline
    rag.save(config["persist_directory"])
    print(f"\n✓ Pipeline saved to {config['persist_directory']}")

    # Load pipeline (fast startup)
    rag_loaded = RAGPipeline(
        embedding_model=config["embedding_model"],
        vector_store_type=config["vector_store"],
        persist_directory=config["persist_directory"],
    )
    rag_loaded.load(config["persist_directory"])
    print("✓ Pipeline loaded from disk (fast startup)")


async def main():
    """Run all examples"""

    print("\n" + "=" * 70)
    print("SciAgent RAG Examples")
    print("=" * 70)

    # Run examples
    await example_1_basic_rag()
    await example_2_graph_rag()
    await example_3_hybrid_rag()
    await example_4_rag_agent()
    await example_5_sciagent_with_rag()
    await example_6_production_rag_pipeline()

    print("\n" + "=" * 70)
    print("All RAG Examples Complete!")
    print("=" * 70)

    print("\n📚 Key Takeaways:")
    print("  1. Traditional RAG: Fast vector-based retrieval")
    print("  2. Graph RAG: Relationship-aware retrieval")
    print("  3. Hybrid RAG: Best of both approaches")
    print("  4. RAG Agent: Easy integration with SciAgent")
    print("  5. Production: Scalable, persistent, fast")

    print("\n🚀 Next Steps:")
    print("  - Load your own documents")
    print("  - Customize embedding models")
    print("  - Fine-tune chunk sizes")
    print("  - Integrate with experiments")
    print("  - Deploy to production")


if __name__ == "__main__":
    asyncio.run(main())
