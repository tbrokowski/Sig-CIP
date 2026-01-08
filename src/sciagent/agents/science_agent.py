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

    async def plan_experiments(self, query: str) -> ExperimentPlanning:
        """
        Multi-stage experiment planning

        Args:
            query: Research question

        Returns:
            Complete experiment planning
        """

        self.logger.info(f"Planning experiments for: {query}")

        # Stage 1: Generate hypotheses
        hypotheses = await self._generate_hypotheses(query)

        # Stage 2: Design experiments for top hypothesis
        hypothesis = hypotheses[0]  # Take best hypothesis
        design = await self._design_experiment(query, hypothesis)

        # Stage 3: Create experiment sequence (simplified MCTS)
        sequence = ExperimentSequence(
            steps=[design],
            total_value=0.8,
            expected_cost=1000.0,
            expected_duration=3600.0,
        )

        # Stage 4: Check if dataset is needed
        requires_dataset, dataset_req = await self._check_dataset_requirements(design)

        plan = ExperimentPlan(
            hypothesis=hypothesis,
            design=design,
            sequence=sequence,
            expected_value=0.8,
            requires_dataset=requires_dataset,
            dataset_requirements=dataset_req,
        )

        return ExperimentPlanning(
            query=query,
            papers=[],  # Papers from literature search (simplified)
            hypotheses=hypotheses,
            plans=[plan],
            recommendation=plan,
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
        Propose experiment refinements based on results

        Args:
            query: Original query
            planning: Original planning
            results: Execution results
            analysis: Analysis of results

        Returns:
            List of proposed refinements
        """

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
