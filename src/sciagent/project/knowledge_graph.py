"""
Scientific Knowledge Graph

Stores and queries scientific knowledge including:
- Papers and their relationships
- Hypotheses and supporting evidence
- Experimental results
- Concept relationships
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from sciagent.utils.logging import logger
from sciagent.utils.models import Hypothesis, Paper


@dataclass
class Concept:
    """A scientific concept"""

    id: str
    name: str
    description: str
    category: str  # "method", "dataset", "metric", "hypothesis", etc.
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Evidence supporting or refuting a hypothesis"""

    source: str  # Paper ID or experiment ID
    supports: bool
    strength: float  # 0.0 to 1.0
    description: str


class ScientificKnowledgeGraph:
    """
    Knowledge graph for storing and querying scientific knowledge

    Uses NetworkX for graph operations with nodes representing:
    - Papers
    - Hypotheses
    - Concepts
    - Experimental results

    Edges represent relationships:
    - "cites" (paper -> paper)
    - "supports" (paper -> hypothesis)
    - "refutes" (paper -> hypothesis)
    - "uses" (paper -> concept)
    - "related_to" (concept -> concept)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize knowledge graph

        Args:
            cache_dir: Directory for caching graph
        """
        self.graph = nx.DiGraph()
        self.cache_dir = cache_dir or Path.home() / ".sciagent" / "knowledge"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing graph if available
        self._load_from_cache()

    def add_paper(self, paper: Paper) -> None:
        """
        Add paper to knowledge graph

        Args:
            paper: Paper to add
        """
        self.graph.add_node(
            paper.id,
            type="paper",
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            year=paper.year,
            citations=paper.citations,
            key_findings=paper.key_findings,
            url=paper.url,
        )

        logger.debug(f"Added paper to knowledge graph: {paper.title}")

    async def add_papers(self, papers: List[Paper]) -> None:
        """
        Add multiple papers to knowledge graph

        Args:
            papers: List of papers to add
        """
        for paper in papers:
            self.add_paper(paper)

        # Extract and link concepts
        await self._extract_concepts(papers)

        logger.info(f"Added {len(papers)} papers to knowledge graph")

    async def _extract_concepts(self, papers: List[Paper]) -> None:
        """
        Extract concepts from papers and create relationships

        Args:
            papers: Papers to extract concepts from
        """
        # Simple keyword-based concept extraction
        # In production, this would use NLP/LLM
        common_concepts = {
            "dropout": "method",
            "batch normalization": "method",
            "resnet": "method",
            "transformer": "method",
            "cifar-10": "dataset",
            "imagenet": "dataset",
            "accuracy": "metric",
            "precision": "metric",
            "recall": "metric",
            "f1-score": "metric",
        }

        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()

            for concept_name, category in common_concepts.items():
                if concept_name in text:
                    # Add concept node if not exists
                    concept_id = f"concept:{concept_name}"

                    if not self.graph.has_node(concept_id):
                        self.graph.add_node(
                            concept_id,
                            type="concept",
                            name=concept_name,
                            category=category,
                        )

                    # Link paper to concept
                    self.graph.add_edge(paper.id, concept_id, relation="uses")

    def add_hypothesis(
        self, hypothesis: Hypothesis, related_papers: Optional[List[str]] = None
    ) -> str:
        """
        Add hypothesis to knowledge graph

        Args:
            hypothesis: Hypothesis to add
            related_papers: IDs of related papers

        Returns:
            Hypothesis ID
        """
        hypothesis_id = f"hypothesis:{hash(hypothesis.statement) % 10000000}"

        self.graph.add_node(
            hypothesis_id,
            type="hypothesis",
            statement=hypothesis.statement,
            rationale=hypothesis.rationale,
            predictions=hypothesis.testable_predictions,
            confidence=hypothesis.confidence,
        )

        # Link to related papers
        if related_papers:
            for paper_id in related_papers:
                if self.graph.has_node(paper_id):
                    self.graph.add_edge(paper_id, hypothesis_id, relation="supports")

        logger.debug(f"Added hypothesis to knowledge graph: {hypothesis.statement}")

        return hypothesis_id

    async def update(
        self, hypothesis: Hypothesis, results: Any, analysis: Any
    ) -> None:
        """
        Update knowledge graph with experimental results

        Args:
            hypothesis: Tested hypothesis
            results: Experimental results
            analysis: Analysis of results
        """
        hypothesis_id = self.add_hypothesis(hypothesis)

        # Add experiment result
        result_id = f"result:{hash(str(results)) % 10000000}"

        self.graph.add_node(
            result_id,
            type="result",
            summary=analysis.summary if hasattr(analysis, "summary") else str(analysis),
            supports_hypothesis=analysis.supports_hypothesis
            if hasattr(analysis, "supports_hypothesis")
            else False,
            confidence=analysis.confidence if hasattr(analysis, "confidence") else 0.5,
        )

        # Link result to hypothesis
        relation = (
            "supports" if getattr(analysis, "supports_hypothesis", False) else "refutes"
        )
        self.graph.add_edge(result_id, hypothesis_id, relation=relation)

        logger.info(f"Updated knowledge graph with experiment result")

        # Save to cache
        self._save_to_cache()

    def query_related_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query for related papers

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of related papers
        """
        results = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "paper":
                # Simple text matching
                text = f"{data.get('title', '')} {data.get('abstract', '')}".lower()
                if query.lower() in text:
                    results.append(
                        {
                            "id": node_id,
                            "title": data.get("title"),
                            "relevance": self._calculate_relevance(node_id, query),
                            **data,
                        }
                    )

        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)

        return results[:limit]

    def _calculate_relevance(self, node_id: str, query: str) -> float:
        """Calculate relevance score for a node"""
        # Simple relevance based on:
        # - Text matching
        # - Citation count
        # - Recency
        # - Graph centrality

        data = self.graph.nodes[node_id]

        # Text match score
        text = f"{data.get('title', '')} {data.get('abstract', '')}".lower()
        text_score = text.count(query.lower()) / max(len(text.split()), 1)

        # Citation score (normalized)
        citation_score = min(data.get("citations", 0) / 1000, 1.0)

        # Recency score (prefer recent papers)
        year = data.get("year", 2000)
        recency_score = (year - 2000) / 24  # Normalize 2000-2024

        # Graph centrality
        try:
            centrality = nx.degree_centrality(self.graph)[node_id]
        except:
            centrality = 0.0

        # Weighted combination
        relevance = (
            0.4 * text_score + 0.3 * citation_score + 0.2 * recency_score + 0.1 * centrality
        )

        return relevance

    def find_supporting_evidence(
        self, hypothesis: Hypothesis
    ) -> Tuple[List[Evidence], List[Evidence]]:
        """
        Find evidence supporting or refuting a hypothesis

        Args:
            hypothesis: Hypothesis to check

        Returns:
            Tuple of (supporting evidence, refuting evidence)
        """
        hypothesis_id = f"hypothesis:{hash(hypothesis.statement) % 10000000}"

        supporting = []
        refuting = []

        # Find nodes connected to hypothesis
        if self.graph.has_node(hypothesis_id):
            for predecessor in self.graph.predecessors(hypothesis_id):
                edge_data = self.graph.edges[predecessor, hypothesis_id]
                relation = edge_data.get("relation")

                node_data = self.graph.nodes[predecessor]

                evidence = Evidence(
                    source=predecessor,
                    supports=(relation == "supports"),
                    strength=0.8,  # Could be calculated from node properties
                    description=node_data.get("title", node_data.get("summary", "")),
                )

                if relation == "supports":
                    supporting.append(evidence)
                elif relation == "refutes":
                    refuting.append(evidence)

        return supporting, refuting

    def get_related_concepts(
        self, concept_name: str, max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept

        Args:
            concept_name: Name of concept
            max_depth: Maximum relationship depth

        Returns:
            List of related concepts
        """
        concept_id = f"concept:{concept_name}"

        if not self.graph.has_node(concept_id):
            return []

        related = []

        # BFS to find related concepts
        visited = set()
        queue = [(concept_id, 0)]

        while queue:
            node_id, depth = queue.pop(0)

            if depth > max_depth or node_id in visited:
                continue

            visited.add(node_id)

            # Get neighbors
            for neighbor in self.graph.neighbors(node_id):
                if self.graph.nodes[neighbor].get("type") == "concept":
                    related.append(
                        {
                            "id": neighbor,
                            "depth": depth + 1,
                            **self.graph.nodes[neighbor],
                        }
                    )
                    queue.append((neighbor, depth + 1))

        return related

    def get_statistics(self) -> Dict[str, int]:
        """Get knowledge graph statistics"""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }

        # Count by type
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        stats.update(type_counts)

        return stats

    def visualize(self, output_path: Path) -> None:
        """
        Visualize knowledge graph

        Args:
            output_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Color by node type
            color_map = {
                "paper": "lightblue",
                "hypothesis": "lightgreen",
                "concept": "lightyellow",
                "result": "lightcoral",
            }

            colors = [
                color_map.get(self.graph.nodes[node].get("type", "unknown"), "gray")
                for node in self.graph.nodes()
            ]

            # Draw graph
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
            nx.draw(
                self.graph,
                pos,
                node_color=colors,
                with_labels=False,
                node_size=300,
                alpha=0.7,
                edge_color="gray",
                arrows=True,
            )

            plt.title("Scientific Knowledge Graph")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Knowledge graph visualization saved to {output_path}")

        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")

    def _save_to_cache(self) -> None:
        """Save graph to cache"""
        cache_file = self.cache_dir / "knowledge_graph.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.graph, f)
            logger.debug("Saved knowledge graph to cache")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    def _load_from_cache(self) -> None:
        """Load graph from cache"""
        cache_file = self.cache_dir / "knowledge_graph.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(
                    f"Loaded knowledge graph from cache: {self.get_statistics()}"
                )
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {e}")
                self.graph = nx.DiGraph()
        else:
            logger.debug("No cached knowledge graph found, starting fresh")

    def export_to_json(self, output_path: Path) -> None:
        """Export graph to JSON format"""
        data = nx.node_link_data(self.graph)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported knowledge graph to {output_path}")

    def import_from_json(self, input_path: Path) -> None:
        """Import graph from JSON format"""
        with open(input_path) as f:
            data = json.load(f)

        self.graph = nx.node_link_graph(data)

        logger.info(f"Imported knowledge graph from {input_path}")
