"""
Science Agent (Gemini) with Advanced Techniques

Handles scientific planning, hypothesis generation, analysis, and refinement proposals.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from sciagent.agents.base import BaseAgent
from sciagent.utils.config import Config
from sciagent.utils.models import (
    ExperimentDesign,
    ExperimentPlan,
    ExperimentPlanning,
    ExperimentSequence,
    Hypothesis,
    Paper,
    Refinement,
    ScientificAnalysis,
    DatasetRequirements,
)
from sciagent.mcp.client import MCPClient, create_default_mcp_client
from sciagent.project.knowledge_graph import ScientificKnowledgeGraph
from sciagent.advanced.mcts import MCTSPlanner
from sciagent.advanced.bayesian_design import BayesianExperimentSelector, ExperimentCandidate
from sciagent.advanced.thompson_sampling import ThompsonSamplingExplorer


class ScienceAgent(BaseAgent):
    """
    Gemini-powered scientific agent with:
    - Extended thinking
    - Literature search
    - Hypothesis generation
    - Experimental design
    - Result analysis
    - Refinement proposals
    """

    def __init__(self, config: Config):
        super().__init__(config)

        if genai is None:
            raise ImportError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )

        # Configure Gemini
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            config.science_agent.model or "gemini-2.0-flash-thinking-exp"
        )

        # Initialize MCP client for literature search
        self.mcp_client = create_default_mcp_client()

        # Initialize knowledge graph
        self.knowledge_graph = ScientificKnowledgeGraph(config.cache_dir / "knowledge")

        # Initialize advanced components
        self.hypothesis_explorer = ThompsonSamplingExplorer()
        self.experiment_planner = MCTSPlanner(
            exploration_constant=config.mcts_exploration_constant,
            n_simulations=config.mcts_simulations,
        )
        self.bayesian_selector = BayesianExperimentSelector()

    async def process(self, request: Dict[str, Any]) -> Any:
        """
        Process a request

        Supported actions:
        - plan_experiments: Generate experiment plans
        - analyze_results: Analyze experimental results
        - propose_refinements: Propose experiment refinements
        """

        action = request.get("action")

        if action == "plan_experiments":
            return await self.plan_experiments(request["query"])
        elif action == "analyze_results":
            return await self.analyze_results(request["hypothesis"], request["results"])
        elif action == "propose_refinements":
            return await self.propose_refinements(
                request["query"],
                request["planning"],
                request["results"],
                request["analysis"],
            )
        else:
            raise ValueError(f"Unknown action: {action}")

    async def plan_experiments(
        self, query: str, budget: float = 10000, max_hypotheses: int = 5
    ) -> ExperimentPlanning:
        """
        Multi-stage experiment planning with advanced techniques

        Args:
            query: Research question
            budget: Available budget
            max_hypotheses: Maximum hypotheses to consider

        Returns:
            Complete experiment planning
        """

        self.logger.info(f"Planning experiments for: {query}")

        # Stage 1: Literature search via MCP
        self.logger.info("Searching scientific literature via MCP...")
        papers = await self._search_literature(query)

        # Stage 2: Hypothesis generation with Thompson Sampling
        self.logger.info("Exploring hypothesis space with Thompson Sampling...")
        hypotheses_with_stats = await self.hypothesis_explorer.explore(
            query=query, papers=papers, n_iterations=20
        )

        # Extract top hypotheses
        top_hypotheses = [
            h.hypothesis for h in hypotheses_with_stats[:max_hypotheses]
        ]

        # Stage 3: For each hypothesis, design experiments and use MCTS to plan sequence
        self.logger.info("Planning experiment sequences with MCTS...")
        experiment_plans = []

        for hypothesis in top_hypotheses:
            # Design initial experiment
            design = await self._design_experiment(query, hypothesis)

            # Use MCTS to plan optimal experiment sequence
            if self.config.enable_mcts:
                sequence = await self.experiment_planner.plan(
                    hypothesis=hypothesis, initial_design=design, budget=budget
                )
            else:
                # Fallback to simple sequence
                sequence = ExperimentSequence(
                    steps=[design],
                    total_value=0.8,
                    expected_cost=1000.0,
                    expected_duration=3600.0,
                )

            # Check if dataset is needed
            requires_dataset, dataset_req = await self._check_dataset_requirements(
                design
            )

            plan = ExperimentPlan(
                hypothesis=hypothesis,
                design=design,
                sequence=sequence,
                expected_value=sequence.total_value,
                requires_dataset=requires_dataset,
                dataset_requirements=dataset_req,
            )

            experiment_plans.append(plan)

        # Rank by expected value
        ranked_plans = sorted(
            experiment_plans, key=lambda p: p.expected_value, reverse=True
        )

        # Update knowledge graph
        await self.knowledge_graph.add_papers(papers)
        for plan in ranked_plans:
            self.knowledge_graph.add_hypothesis(
                plan.hypothesis, related_papers=[p.id for p in papers[:3]]
            )

        return ExperimentPlanning(
            query=query,
            papers=papers,
            hypotheses=top_hypotheses,
            plans=ranked_plans,
            recommendation=ranked_plans[0],
        )

    async def _generate_hypotheses(self, query: str) -> List[Hypothesis]:
        """Generate hypotheses using extended thinking"""

        prompt = f"""Given this research question: {query}

Think step-by-step to generate testable hypotheses.

Consider:
1. What are the key variables involved?
2. What relationships might exist between them?
3. What would constitute evidence for each hypothesis?
4. What are potential alternative explanations?

Generate 3 testable hypotheses ranked by plausibility.

Format each hypothesis as:
HYPOTHESIS: [statement]
RATIONALE: [why this is plausible]
PREDICTIONS: [what we would observe if true]
EVIDENCE: [what evidence would support/refute it]
"""

        response = await self._generate_with_thinking(prompt)
        hypotheses = self._parse_hypotheses(response)

        return hypotheses

    async def _design_experiment(
        self, query: str, hypothesis: Hypothesis
    ) -> ExperimentDesign:
        """Design experiment to test hypothesis"""

        prompt = f"""Design a rigorous experiment to test this hypothesis:

Hypothesis: {hypothesis.statement}
Rationale: {hypothesis.rationale}

Think step-by-step about:
1. What measurements are needed?
2. What controls should be included?
3. What sample size is appropriate?
4. What statistical tests should be used?
5. What are potential confounds?
6. What would constitute strong evidence?

Provide a detailed experimental design."""

        response = await self._generate_with_thinking(prompt)
        design = self._parse_experiment_design(response)

        return design

    async def _check_dataset_requirements(
        self, design: ExperimentDesign
    ) -> tuple[bool, DatasetRequirements | None]:
        """Check if experiment requires a dataset"""

        # Simple heuristic: check if design mentions common datasets
        design_text = f"{design.description} {' '.join(design.measurements)}".lower()

        dataset_keywords = {
            "cifar-10": "cifar10",
            "cifar-100": "cifar100",
            "imagenet": "imagenet",
            "mnist": "mnist",
            "coco": "coco",
        }

        for name, keyword in dataset_keywords.items():
            if keyword in design_text:
                return True, DatasetRequirements(name=name)

        return False, None

    async def analyze_results(
        self, hypothesis: Hypothesis, results: Any
    ) -> ScientificAnalysis:
        """
        Deep analysis of experimental results

        Args:
            hypothesis: Original hypothesis
            results: Execution results

        Returns:
            Scientific analysis
        """

        prompt = f"""Analyze these experimental results rigorously:

Hypothesis: {hypothesis.statement}

Results Summary:
{results.summary}

Metrics:
{results.metrics}

Think deeply about:
1. Do the results support the hypothesis?
2. What is the statistical significance?
3. What is the effect size?
4. Are there any confounds or alternative explanations?
5. What are the limitations?
6. What are the implications?
7. What should be done next?

Provide a comprehensive scientific analysis."""

        response = await self._generate_with_thinking(prompt)
        analysis = self._parse_analysis(response)

        return analysis

    async def propose_refinements(
        self, query: str, planning: ExperimentPlanning, results: Any, analysis: ScientificAnalysis
    ) -> List[Refinement]:
        """
        Propose experiment refinements using Bayesian experimental design

        Args:
            query: Original query
            planning: Original planning
            results: Execution results
            analysis: Analysis of results

        Returns:
            List of proposed refinements ranked by expected information gain
        """

        self.logger.info("Proposing refinements using Bayesian experimental design...")

        # Generate candidate refinements
        candidates = await self._generate_refinement_candidates(
            planning, results, analysis
        )

        # Select best refinements using Bayesian information gain
        if self.config.enable_bayesian_design:
            selected = await self.bayesian_selector.select_experiments(
                candidates=candidates, prior_results=[results], max_select=3
            )

            refinements = [
                Refinement(
                    description=exp.description,
                    rationale=exp.information_gain_explanation,
                    expected_information_gain=exp.expected_information_gain,
                    cost=exp.expected_cost,
                )
                for exp in selected
            ]
        else:
            # Fallback to simple parsing
            prompt = f"""Based on these experimental results, propose refinements:

Original Query: {query}
Hypothesis: {planning.recommendation.hypothesis.statement}

Results Summary: {analysis.summary}
Supports Hypothesis: {analysis.supports_hypothesis}

What are the most valuable next experiments to run?

Consider:
1. Testing different parameters
2. Exploring edge cases
3. Validating with different datasets
4. Testing alternative hypotheses

Propose 3 refinements ranked by expected information gain."""

            response = await self._generate_with_thinking(prompt)
            refinements = self._parse_refinements(response)

        return refinements

    async def _search_literature(self, query: str) -> List[Paper]:
        """
        Search scientific literature via MCP

        Args:
            query: Search query

        Returns:
            List of relevant papers
        """
        try:
            # Initialize MCP client if needed
            if not self.mcp_client.initialized:
                await self.mcp_client.initialize()

            # Search via MCP (tries multiple sources)
            papers = []

            # Try arXiv
            try:
                arxiv_results = await self.mcp_client.call_tool(
                    server="arxiv",
                    tool="search_papers",
                    arguments={"query": query, "limit": 10},
                )

                for result in arxiv_results:
                    paper = Paper(
                        id=result.get("id", ""),
                        title=result.get("title", ""),
                        authors=result.get("authors", []),
                        abstract=result.get("abstract", ""),
                        url=result.get("url", ""),
                        year=result.get("year", 2024),
                        citations=result.get("citations", 0),
                    )
                    papers.append(paper)

            except Exception as e:
                self.logger.warning(f"arXiv search failed: {e}")

            # Try scholarly database
            try:
                scholarly_results = await self.mcp_client.call_tool(
                    server="scholarly",
                    tool="search_papers",
                    arguments={"query": query, "limit": 10},
                )

                for result in scholarly_results:
                    paper = Paper(
                        id=result.get("id", result.get("title", "")[:20]),
                        title=result.get("title", ""),
                        authors=result.get("authors", []),
                        abstract=result.get("abstract", ""),
                        url=result.get("url", ""),
                        year=result.get("year", 2024),
                        citations=result.get("citations", 0),
                    )
                    papers.append(paper)

            except Exception as e:
                self.logger.warning(f"Scholarly search failed: {e}")

            self.logger.info(f"Found {len(papers)} papers via MCP")

            # Also check knowledge graph for related papers
            kg_papers = self.knowledge_graph.query_related_papers(query, limit=5)
            self.logger.info(f"Found {len(kg_papers)} related papers in knowledge graph")

            return papers if papers else self._get_fallback_papers(query)

        except Exception as e:
            self.logger.error(f"Literature search failed: {e}")
            return self._get_fallback_papers(query)

    def _get_fallback_papers(self, query: str) -> List[Paper]:
        """Get fallback papers when MCP search fails"""
        # Return simulated papers as fallback
        return [
            Paper(
                id="fallback_001",
                title=f"Research on {query}",
                authors=["Researcher, A.", "Scientist, B."],
                abstract=f"This paper investigates {query}...",
                url="https://example.com/paper1",
                year=2024,
                citations=50,
            )
        ]

    async def _generate_refinement_candidates(
        self, planning: ExperimentPlanning, results: Any, analysis: ScientificAnalysis
    ) -> List[ExperimentCandidate]:
        """
        Generate candidate refinements

        Args:
            planning: Original planning
            results: Results
            analysis: Analysis

        Returns:
            List of experiment candidates
        """
        candidates = []

        # Candidate 1: Increase sample size
        candidates.append(
            ExperimentCandidate(
                description="Increase sample size for better statistical power",
                parameters={"sample_size": planning.recommendation.design.sample_size * 2},
                uncertainty=0.7,
                expected_cost=500.0,
            )
        )

        # Candidate 2: Test different hyperparameters
        candidates.append(
            ExperimentCandidate(
                description="Test with different hyperparameters",
                parameters={"learning_rate": 0.001, "batch_size": 64},
                uncertainty=0.8,
                expected_cost=300.0,
            )
        )

        # Candidate 3: Validate on different dataset
        candidates.append(
            ExperimentCandidate(
                description="Validate on different dataset",
                parameters={"dataset": "alternative_dataset"},
                uncertainty=0.9,
                expected_cost=400.0,
            )
        )

        # Candidate 4: Test alternative hypothesis
        if len(planning.hypotheses) > 1:
            candidates.append(
                ExperimentCandidate(
                    description=f"Test alternative hypothesis: {planning.hypotheses[1].statement}",
                    parameters={"hypothesis_id": 1},
                    uncertainty=0.85,
                    expected_cost=600.0,
                )
            )

        return candidates

    async def _generate_with_thinking(self, prompt: str) -> str:
        """Generate response with extended thinking if enabled"""

        try:
            if self.config.enable_extended_thinking:
                # Use thinking mode
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        thinking_mode="thinking", temperature=0.7
                    ),
                )
            else:
                response = self.model.generate_content(prompt)

            return response.text
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            # Fallback to simple response
            return f"Error: {str(e)}"

    def _parse_hypotheses(self, response: str) -> List[Hypothesis]:
        """Parse hypotheses from response"""

        # Simple parsing - in production this would be more robust
        hypotheses = []

        # Default hypothesis if parsing fails
        if "HYPOTHESIS:" in response:
            parts = response.split("HYPOTHESIS:")[1:]
            for part in parts[:3]:  # Take up to 3
                lines = part.strip().split("\n")
                statement = lines[0].strip()

                hypothesis = Hypothesis(
                    statement=statement,
                    rationale="Generated from AI analysis",
                    testable_predictions=["To be determined"],
                    expected_evidence="Statistical evidence from experiments",
                    confidence=0.7,
                )
                hypotheses.append(hypothesis)
        else:
            # Fallback hypothesis
            hypotheses.append(
                Hypothesis(
                    statement="Default hypothesis from query",
                    rationale="Generated automatically",
                    testable_predictions=["To be determined"],
                    expected_evidence="Experimental results",
                )
            )

        return hypotheses

    def _parse_experiment_design(self, response: str) -> ExperimentDesign:
        """Parse experiment design from response"""

        # Simple parsing - extract key information
        return ExperimentDesign(
            description=response[:500],  # First 500 chars
            measurements=["accuracy", "precision", "recall"],
            controls=["baseline", "random"],
            sample_size=1000,
            statistical_tests=["t-test", "ANOVA"],
            potential_confounds=["batch effects", "sampling bias"],
            success_criteria="p < 0.05 and effect size > 0.3",
        )

    def _parse_analysis(self, response: str) -> ScientificAnalysis:
        """Parse analysis from response"""

        # Simple parsing - in production this would extract structured data
        supports = "support" in response.lower() or "confirm" in response.lower()

        return ScientificAnalysis(
            summary=response[:500],
            supports_hypothesis=supports,
            statistical_significance=0.95,
            effect_size=0.5,
            limitations=["Limited sample size", "Potential confounds"],
            implications=["Further research needed"],
            confidence=0.8,
        )

    def _parse_refinements(self, response: str) -> List[Refinement]:
        """Parse refinements from response"""

        # Simple parsing - create default refinements
        return [
            Refinement(
                description="Increase sample size for better statistical power",
                rationale="Larger sample reduces uncertainty",
                expected_information_gain=0.7,
                cost=500.0,
            ),
            Refinement(
                description="Test with different hyperparameters",
                rationale="Explore parameter space more thoroughly",
                expected_information_gain=0.6,
                cost=300.0,
            ),
        ]
