"""
RAG-Enhanced Agent

Combines RAG capabilities with agent workflows for:
- Document-based question answering
- Literature-aware hypothesis generation
- Context-aware code generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from sciagent.agents.base import BaseAgent
from sciagent.rag import RAGPipeline, GraphRAG, HybridRAG
from sciagent.project.knowledge_graph import ScientificKnowledgeGraph
from sciagent.utils.config import Config
from sciagent.utils.logging import logger


class RAGAgent(BaseAgent):
    """
    Agent with RAG capabilities

    Can answer questions using:
    - Document corpus (traditional RAG)
    - Knowledge graph (Graph RAG)
    - Hybrid approach
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # Initialize RAG systems
        persist_dir = config.cache_dir / "rag"
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Traditional RAG
        self.rag = RAGPipeline(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_type="faiss",
            persist_directory=persist_dir / "vector_store",
            chunk_size=512,
            chunk_overlap=50,
        )

        # Knowledge graph
        self.kg = ScientificKnowledgeGraph(config.cache_dir / "knowledge")

        # Graph RAG
        self.graph_rag = GraphRAG(
            knowledge_graph=self.kg,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            max_hops=3,
        )

        # Hybrid RAG
        self.hybrid_rag = HybridRAG(
            rag_pipeline=self.rag,
            graph_rag=self.graph_rag,
            combination_strategy="weighted",
        )

        self.corpus_loaded = False

    async def process(self, request: Dict[str, Any]) -> Any:
        """
        Process a request

        Supported actions:
        - add_documents: Add documents to RAG
        - query: Query RAG system
        - query_graph: Query Graph RAG
        - query_hybrid: Query Hybrid RAG
        """

        action = request.get("action")

        if action == "add_documents":
            return await self.add_documents(
                request["documents"], request.get("metadata")
            )

        elif action == "query":
            return await self.query(
                request["query"],
                top_k=request.get("top_k", 5),
                mode=request.get("mode", "rag"),
            )

        elif action == "query_graph":
            return await self.query_graph(
                request["query"],
                top_k=request.get("top_k", 5),
                use_multi_hop=request.get("use_multi_hop", False),
            )

        elif action == "query_hybrid":
            return await self.query_hybrid(
                request["query"],
                top_k=request.get("top_k", 5),
                rag_weight=request.get("rag_weight", 0.6),
            )

        else:
            raise ValueError(f"Unknown action: {action}")

    async def add_documents(
        self, documents: List[str], metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Add documents to RAG systems

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document

        Returns:
            Status information
        """
        logger.info(f"Adding {len(documents)} documents to RAG systems...")

        # Add to traditional RAG
        self.rag.add_documents(documents, metadata)

        # Add to Graph RAG (through knowledge graph)
        self.graph_rag.add_documents_to_graph(documents)

        self.corpus_loaded = True

        return {
            "status": "success",
            "documents_added": len(documents),
            "total_documents": len(self.rag.vector_store.documents),
        }

    async def query(
        self, query: str, top_k: int = 5, mode: str = "rag"
    ) -> Dict[str, Any]:
        """
        Query RAG system

        Args:
            query: Query text
            top_k: Number of results
            mode: Mode (rag, graph, hybrid)

        Returns:
            Query results with context
        """
        if not self.corpus_loaded:
            logger.warning("No documents loaded in RAG system")

        if mode == "rag":
            results, context = self.rag.query(query, top_k=top_k)
            return {
                "query": query,
                "mode": "rag",
                "results": [
                    {
                        "content": r.document.content,
                        "score": r.score,
                        "rank": r.rank,
                        "metadata": r.document.metadata,
                    }
                    for r in results
                ],
                "context": context,
            }

        elif mode == "graph":
            return await self.query_graph(query, top_k=top_k)

        elif mode == "hybrid":
            return await self.query_hybrid(query, top_k=top_k)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    async def query_graph(
        self, query: str, top_k: int = 5, use_multi_hop: bool = False
    ) -> Dict[str, Any]:
        """
        Query Graph RAG

        Args:
            query: Query text
            top_k: Number of results
            use_multi_hop: Use multi-hop retrieval

        Returns:
            Graph RAG results
        """
        results, context = self.graph_rag.query(
            query, top_k=top_k, use_multi_hop=use_multi_hop
        )

        return {
            "query": query,
            "mode": "graph",
            "results": [
                {
                    "content": r.node.content,
                    "score": r.score,
                    "rank": r.rank,
                    "node_type": r.node.node_type,
                    "node_id": r.node.node_id,
                    "paths": r.paths,
                    "related_nodes": [
                        {"type": n.node_type, "content": n.content[:100]}
                        for n in r.related_nodes
                    ],
                }
                for r in results
            ],
            "context": context,
        }

    async def query_hybrid(
        self, query: str, top_k: int = 5, rag_weight: float = 0.6
    ) -> Dict[str, Any]:
        """
        Query Hybrid RAG

        Args:
            query: Query text
            top_k: Number of results
            rag_weight: Weight for RAG vs Graph RAG

        Returns:
            Hybrid results
        """
        results, context = self.hybrid_rag.query(
            query, top_k=top_k, rag_weight=rag_weight
        )

        return {
            "query": query,
            "mode": "hybrid",
            "rag_weight": rag_weight,
            "results": [
                {
                    "content": (
                        r.document.content
                        if hasattr(r, "document")
                        else r.node.content
                    ),
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in results
            ],
            "context": context,
        }

    def load_corpus_from_directory(
        self, directory: Path, file_pattern: str = "*.txt"
    ) -> None:
        """
        Load corpus from directory

        Args:
            directory: Directory containing documents
            file_pattern: File pattern to match
        """
        logger.info(f"Loading corpus from {directory}...")

        documents = []
        metadata = []

        for file_path in directory.glob(file_pattern):
            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append(content)
                metadata.append({"source": str(file_path), "filename": file_path.name})
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        if documents:
            import asyncio

            asyncio.create_task(self.add_documents(documents, metadata))
            logger.info(f"Loaded {len(documents)} documents from {directory}")
        else:
            logger.warning(f"No documents found in {directory}")

    def save(self, path: Path) -> None:
        """Save RAG systems"""
        path.mkdir(parents=True, exist_ok=True)

        # Save RAG pipeline
        self.rag.save(path / "rag")

        # Save knowledge graph
        self.kg._save_to_cache()

        logger.info(f"Saved RAG agent to {path}")

    def load(self, path: Path) -> None:
        """Load RAG systems"""
        if (path / "rag").exists():
            self.rag.load(path / "rag")
            self.corpus_loaded = True

        # Knowledge graph auto-loads
        self.kg._load_from_cache()

        # Rebuild graph RAG embeddings
        self.graph_rag._build_node_embeddings()

        logger.info(f"Loaded RAG agent from {path}")


def create_rag_enhanced_response(
    base_response: str, rag_context: str, citations: bool = True
) -> str:
    """
    Create RAG-enhanced response with citations

    Args:
        base_response: Base LLM response
        rag_context: Retrieved context
        citations: Whether to add citations

    Returns:
        Enhanced response
    """
    if citations:
        enhanced = f"""{base_response}

---
Context Sources:
{rag_context}
"""
    else:
        enhanced = base_response

    return enhanced
