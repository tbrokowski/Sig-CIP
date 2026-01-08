"""
Graph RAG - Combining Knowledge Graphs with RAG

Implements Graph RAG which:
- Uses knowledge graph structure for retrieval
- Combines semantic search with graph traversal
- Multi-hop reasoning over knowledge
- Relationship-aware context generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from sciagent.project.knowledge_graph import ScientificKnowledgeGraph
from sciagent.rag.pipeline import Document, EmbeddingModel, RetrievalResult
from sciagent.utils.logging import logger


@dataclass
class GraphNode:
    """A node in the graph with embedding"""

    node_id: str
    node_type: str
    content: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class GraphRetrievalResult:
    """Result from graph retrieval"""

    node: GraphNode
    score: float
    rank: int
    paths: List[List[str]]  # Paths from query to this node
    related_nodes: List[GraphNode]  # Connected nodes


class GraphRAG:
    """
    Graph RAG combines knowledge graph with RAG

    Features:
    - Graph structure-aware retrieval
    - Multi-hop reasoning
    - Relationship context
    - Path-based explanations
    """

    def __init__(
        self,
        knowledge_graph: ScientificKnowledgeGraph,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_hops: int = 3,
    ):
        """
        Initialize Graph RAG

        Args:
            knowledge_graph: Scientific knowledge graph
            embedding_model: Embedding model name
            max_hops: Maximum hops for graph traversal
        """
        self.kg = knowledge_graph
        self.embedding_model = EmbeddingModel(embedding_model)
        self.max_hops = max_hops

        # Cache of node embeddings
        self.node_embeddings: Dict[str, np.ndarray] = {}

        # Build embeddings for existing nodes
        self._build_node_embeddings()

    def _build_node_embeddings(self) -> None:
        """Build embeddings for all nodes in knowledge graph"""
        logger.info("Building embeddings for knowledge graph nodes...")

        nodes_to_embed = []
        node_ids = []

        for node_id, data in self.kg.graph.nodes(data=True):
            # Extract text content from node
            content = self._extract_node_content(node_id, data)
            if content:
                nodes_to_embed.append(content)
                node_ids.append(node_id)

        if nodes_to_embed:
            # Batch embed
            embeddings = self.embedding_model.embed(nodes_to_embed)

            # Store embeddings
            for node_id, embedding in zip(node_ids, embeddings):
                self.node_embeddings[node_id] = embedding

            logger.info(f"Built embeddings for {len(node_ids)} nodes")

    def _extract_node_content(self, node_id: str, data: Dict[str, Any]) -> str:
        """Extract text content from a node"""
        node_type = data.get("type", "unknown")

        if node_type == "paper":
            return f"{data.get('title', '')} {data.get('abstract', '')}"
        elif node_type == "hypothesis":
            return data.get("statement", "")
        elif node_type == "concept":
            return data.get("name", "")
        elif node_type == "result":
            return data.get("summary", "")
        else:
            return str(data)

    def retrieve(
        self, query: str, top_k: int = 5, include_neighbors: bool = True
    ) -> List[GraphRetrievalResult]:
        """
        Retrieve relevant nodes from knowledge graph

        Args:
            query: Query text
            top_k: Number of results
            include_neighbors: Include neighboring nodes

        Returns:
            Graph retrieval results
        """
        # Embed query
        query_embedding = self.embedding_model.embed(query)[0]

        # Compute similarity with all nodes
        similarities = []
        node_ids = []

        for node_id, embedding in self.node_embeddings.items():
            # Cosine similarity
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append(sim)
            node_ids.append(node_id)

        # Get top-k
        if len(similarities) == 0:
            return []

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices):
            node_id = node_ids[idx]
            node_data = self.kg.graph.nodes[node_id]

            # Create graph node
            graph_node = GraphNode(
                node_id=node_id,
                node_type=node_data.get("type", "unknown"),
                content=self._extract_node_content(node_id, node_data),
                properties=node_data,
                embedding=self.node_embeddings[node_id],
            )

            # Find paths to this node (for explanation)
            paths = self._find_paths_to_node(query, node_id)

            # Get related nodes
            related = []
            if include_neighbors:
                related = self._get_related_nodes(node_id)

            result = GraphRetrievalResult(
                node=graph_node,
                score=similarities[idx],
                rank=rank,
                paths=paths,
                related_nodes=related,
            )

            results.append(result)

        return results

    def _find_paths_to_node(
        self, query: str, target_node: str, max_paths: int = 3
    ) -> List[List[str]]:
        """
        Find paths from query-related nodes to target node

        Args:
            query: Query text
            target_node: Target node ID
            max_paths: Maximum paths to return

        Returns:
            List of paths (each path is list of node IDs)
        """
        import networkx as nx

        paths = []

        # Find nodes most related to query
        query_embedding = self.embedding_model.embed(query)[0]

        top_nodes = []
        for node_id, embedding in self.node_embeddings.items():
            if node_id != target_node:
                sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                top_nodes.append((node_id, sim))

        top_nodes.sort(key=lambda x: x[1], reverse=True)

        # Try to find paths from top nodes to target
        for source_node, _ in top_nodes[:5]:
            try:
                # Find shortest path
                if nx.has_path(self.kg.graph, source_node, target_node):
                    path = nx.shortest_path(self.kg.graph, source_node, target_node)
                    if len(path) <= self.max_hops + 1:
                        paths.append(path)

                if len(paths) >= max_paths:
                    break
            except:
                continue

        return paths

    def _get_related_nodes(self, node_id: str, max_neighbors: int = 5) -> List[GraphNode]:
        """
        Get related nodes (neighbors in graph)

        Args:
            node_id: Node ID
            max_neighbors: Maximum neighbors to return

        Returns:
            List of related graph nodes
        """
        related = []

        # Get neighbors
        neighbors = list(self.kg.graph.neighbors(node_id))[:max_neighbors]

        for neighbor_id in neighbors:
            neighbor_data = self.kg.graph.nodes[neighbor_id]

            graph_node = GraphNode(
                node_id=neighbor_id,
                node_type=neighbor_data.get("type", "unknown"),
                content=self._extract_node_content(neighbor_id, neighbor_data),
                properties=neighbor_data,
                embedding=self.node_embeddings.get(neighbor_id),
            )

            related.append(graph_node)

        return related

    def multi_hop_retrieve(
        self, query: str, num_hops: int = 2, top_k_per_hop: int = 3
    ) -> List[GraphRetrievalResult]:
        """
        Multi-hop retrieval for complex queries

        Args:
            query: Query text
            num_hops: Number of hops
            top_k_per_hop: Nodes to retrieve per hop

        Returns:
            Results from multi-hop retrieval
        """
        logger.info(f"Performing {num_hops}-hop retrieval...")

        # First hop - direct retrieval
        current_results = self.retrieve(query, top_k=top_k_per_hop, include_neighbors=False)

        all_results = list(current_results)
        visited_nodes = {r.node.node_id for r in current_results}

        # Subsequent hops - traverse from current nodes
        for hop in range(1, num_hops):
            next_nodes = set()

            # Collect neighbors from current hop
            for result in current_results:
                neighbors = list(self.kg.graph.neighbors(result.node.node_id))
                next_nodes.update(n for n in neighbors if n not in visited_nodes)

            if not next_nodes:
                break

            # Rank next nodes by relevance to query
            query_embedding = self.embedding_model.embed(query)[0]
            node_scores = []

            for node_id in next_nodes:
                if node_id in self.node_embeddings:
                    embedding = self.node_embeddings[node_id]
                    sim = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    node_scores.append((node_id, sim))

            node_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top-k for next hop
            current_results = []
            for rank, (node_id, score) in enumerate(node_scores[:top_k_per_hop]):
                node_data = self.kg.graph.nodes[node_id]

                graph_node = GraphNode(
                    node_id=node_id,
                    node_type=node_data.get("type", "unknown"),
                    content=self._extract_node_content(node_id, node_data),
                    properties=node_data,
                    embedding=self.node_embeddings[node_id],
                )

                result = GraphRetrievalResult(
                    node=graph_node,
                    score=score,
                    rank=rank,
                    paths=[],
                    related_nodes=[],
                )

                current_results.append(result)
                all_results.append(result)
                visited_nodes.add(node_id)

        logger.info(f"Multi-hop retrieval found {len(all_results)} total nodes")

        return all_results

    def query(
        self, query: str, top_k: int = 5, use_multi_hop: bool = False
    ) -> Tuple[List[GraphRetrievalResult], str]:
        """
        Query Graph RAG

        Args:
            query: Query text
            top_k: Number of results
            use_multi_hop: Use multi-hop retrieval

        Returns:
            Results and formatted context
        """
        if use_multi_hop:
            results = self.multi_hop_retrieve(query, num_hops=2, top_k_per_hop=top_k)
        else:
            results = self.retrieve(query, top_k=top_k)

        context = self._format_graph_context(results)

        return results, context

    def _format_graph_context(self, results: List[GraphRetrievalResult]) -> str:
        """Format graph retrieval results as context"""
        context_parts = []

        for i, result in enumerate(results, 1):
            node = result.node

            part = f"[{node.node_type.upper()} {i}] (Score: {result.score:.3f})\n"
            part += f"{node.content}\n"

            # Add relationship information
            if result.related_nodes:
                part += f"\nRelated nodes:\n"
                for rel_node in result.related_nodes[:3]:
                    part += f"  - {rel_node.node_type}: {rel_node.content[:100]}...\n"

            # Add paths if available
            if result.paths:
                part += f"\nPaths from query: {len(result.paths)} path(s) found\n"

            context_parts.append(part)

        return "\n".join(context_parts)

    def add_documents_to_graph(
        self, documents: List[str], doc_type: str = "paper"
    ) -> None:
        """
        Add documents to knowledge graph and update embeddings

        Args:
            documents: Documents to add
            doc_type: Type of documents
        """
        from sciagent.utils.models import Paper

        # Convert to Paper objects and add to KG
        papers = []
        for i, doc in enumerate(documents):
            paper = Paper(
                id=f"doc_{i}",
                title=f"Document {i}",
                authors=["Auto-generated"],
                abstract=doc[:500],
                year=2024,
                citations=0,
            )
            papers.append(paper)

        # Add to knowledge graph
        import asyncio

        asyncio.create_task(self.kg.add_papers(papers))

        # Update embeddings
        self._build_node_embeddings()

        logger.info(f"Added {len(documents)} documents to knowledge graph")

    def hybrid_search(
        self, query: str, top_k: int = 5, alpha: float = 0.7
    ) -> Tuple[List[GraphRetrievalResult], str]:
        """
        Hybrid search combining graph structure and semantic similarity

        Args:
            query: Query text
            top_k: Number of results
            alpha: Weight for semantic similarity (1-alpha for graph structure)

        Returns:
            Hybrid results and context
        """
        # Semantic search
        semantic_results = self.retrieve(query, top_k=top_k * 2, include_neighbors=False)

        # Graph structure scoring (PageRank)
        import networkx as nx

        try:
            pagerank = nx.pagerank(self.kg.graph)
        except:
            pagerank = {node: 1.0 for node in self.kg.graph.nodes()}

        # Combine scores
        for result in semantic_results:
            structure_score = pagerank.get(result.node.node_id, 0.0)
            result.score = alpha * result.score + (1 - alpha) * structure_score

        # Re-sort
        semantic_results.sort(key=lambda r: r.score, reverse=True)

        # Update ranks
        for rank, result in enumerate(semantic_results[:top_k]):
            result.rank = rank

        results = semantic_results[:top_k]
        context = self._format_graph_context(results)

        return results, context


class HybridRAG:
    """
    Hybrid RAG combining traditional RAG with Graph RAG

    Uses both vector similarity and graph structure
    """

    def __init__(
        self, rag_pipeline, graph_rag: GraphRAG, combination_strategy: str = "weighted"
    ):
        """
        Initialize Hybrid RAG

        Args:
            rag_pipeline: Traditional RAG pipeline
            graph_rag: Graph RAG
            combination_strategy: How to combine results (weighted, rerank)
        """
        from sciagent.rag.pipeline import RAGPipeline

        self.rag = rag_pipeline
        self.graph_rag = graph_rag
        self.combination_strategy = combination_strategy

    def query(
        self, query: str, top_k: int = 5, rag_weight: float = 0.6
    ) -> Tuple[List[Any], str]:
        """
        Hybrid query combining RAG and Graph RAG

        Args:
            query: Query text
            top_k: Number of results
            rag_weight: Weight for RAG results (1-rag_weight for Graph RAG)

        Returns:
            Combined results and context
        """
        # Get results from both
        rag_results, rag_context = self.rag.query(query, top_k=top_k)
        graph_results, graph_context = self.graph_rag.query(query, top_k=top_k)

        if self.combination_strategy == "weighted":
            # Combine contexts with weighting
            combined_context = f"""=== RAG Results (weight: {rag_weight}) ===
{rag_context}

=== Graph RAG Results (weight: {1-rag_weight}) ===
{graph_context}
"""

            return rag_results + graph_results, combined_context

        else:  # rerank
            # Re-rank combined results
            all_results = list(rag_results) + list(graph_results)

            # Simple deduplication and ranking
            seen = set()
            unique_results = []

            for result in all_results:
                content = (
                    result.document.content
                    if hasattr(result, "document")
                    else result.node.content
                )
                if content not in seen:
                    seen.add(content)
                    unique_results.append(result)

            combined_context = f"{rag_context}\n\n{graph_context}"

            return unique_results[:top_k], combined_context
