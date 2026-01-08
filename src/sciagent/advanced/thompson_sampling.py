"""
Thompson Sampling for Hypothesis Exploration

Uses Thompson sampling (Bayesian bandit algorithm) to balance exploration
and exploitation when testing multiple hypotheses.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sciagent.utils.logging import logger
from sciagent.utils.models import Hypothesis, Paper


@dataclass
class HypothesisWithStats:
    """Hypothesis with Thompson sampling statistics"""

    hypothesis: Hypothesis
    alpha: float = 1.0  # Beta distribution alpha (successes + 1)
    beta: float = 1.0  # Beta distribution beta (failures + 1)
    samples: int = 0
    mean_reward: float = 0.0

    @property
    def expected_value(self) -> float:
        """Expected value (mean of beta distribution)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Uncertainty (variance of beta distribution)"""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))


class ThompsonSamplingExplorer:
    """
    Thompson sampling for hypothesis exploration

    Explores multiple hypotheses by sampling from posterior distributions,
    balancing between:
    - Exploitation: Testing promising hypotheses
    - Exploration: Testing uncertain hypotheses
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay_factor: float = 0.95,
    ):
        """
        Initialize Thompson sampling explorer

        Args:
            prior_alpha: Prior alpha for beta distribution
            prior_beta: Prior beta for beta distribution
            decay_factor: Decay factor for old observations
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_factor = decay_factor

        self.hypotheses: List[HypothesisWithStats] = []
        self.iteration = 0

    async def explore(
        self,
        query: str,
        papers: List[Paper],
        n_iterations: int = 20,
        hypotheses_per_iteration: int = 5,
    ) -> List[HypothesisWithStats]:
        """
        Explore hypothesis space using Thompson sampling

        Args:
            query: Research query
            papers: Relevant papers
            n_iterations: Number of exploration iterations
            hypotheses_per_iteration: Hypotheses to generate per iteration

        Returns:
            List of hypotheses with statistics
        """
        logger.info(
            f"Exploring hypothesis space with Thompson sampling "
            f"({n_iterations} iterations)"
        )

        # Initialize with diverse hypotheses
        initial_hypotheses = await self._generate_initial_hypotheses(
            query, papers, hypotheses_per_iteration * 2
        )

        self.hypotheses = [
            HypothesisWithStats(
                hypothesis=h, alpha=self.prior_alpha, beta=self.prior_beta
            )
            for h in initial_hypotheses
        ]

        # Exploration iterations
        for i in range(n_iterations):
            self.iteration = i

            # Sample hypothesis using Thompson sampling
            selected_hypothesis = self._thompson_sample()

            # Simulate testing hypothesis (in production, would actually test)
            reward = await self._evaluate_hypothesis(
                selected_hypothesis.hypothesis, papers
            )

            # Update statistics
            self._update(selected_hypothesis, reward)

            if (i + 1) % 5 == 0:
                best = max(self.hypotheses, key=lambda h: h.expected_value)
                logger.debug(
                    f"Iteration {i+1}/{n_iterations}: "
                    f"Best hypothesis EV={best.expected_value:.3f}, "
                    f"Uncertainty={best.uncertainty:.3f}"
                )

        # Sort by expected value
        self.hypotheses.sort(key=lambda h: h.expected_value, reverse=True)

        logger.info(
            f"Exploration complete. Top hypothesis: "
            f"{self.hypotheses[0].hypothesis.statement[:50]}..."
        )

        return self.hypotheses

    def _thompson_sample(self) -> HypothesisWithStats:
        """
        Sample hypothesis using Thompson sampling

        Returns:
            Selected hypothesis
        """
        # Sample from each hypothesis's posterior distribution
        samples = []

        for h_stats in self.hypotheses:
            # Sample from beta distribution
            sample = np.random.beta(h_stats.alpha, h_stats.beta)
            samples.append((sample, h_stats))

        # Select hypothesis with highest sample
        selected = max(samples, key=lambda x: x[0])[1]

        return selected

    async def _evaluate_hypothesis(
        self, hypothesis: Hypothesis, papers: List[Paper]
    ) -> float:
        """
        Evaluate hypothesis quality (simulated)

        Args:
            hypothesis: Hypothesis to evaluate
            papers: Relevant papers

        Returns:
            Reward (0-1)
        """
        # Simulated evaluation based on:
        # 1. Novelty
        # 2. Testability
        # 3. Support from papers
        # 4. Clarity

        # Novelty: Check if similar hypotheses exist
        novelty = 1.0 - self._calculate_hypothesis_similarity(hypothesis)

        # Testability: Based on number of predictions
        testability = min(len(hypothesis.testable_predictions) / 5.0, 1.0)

        # Paper support: Check if papers discuss related concepts
        paper_support = self._calculate_paper_support(hypothesis, papers)

        # Clarity: Length and specificity (simple heuristic)
        clarity = min(len(hypothesis.statement) / 100.0, 1.0)

        # Weighted combination
        reward = (
            0.3 * novelty + 0.3 * testability + 0.3 * paper_support + 0.1 * clarity
        )

        # Add noise to simulate uncertainty
        noise = np.random.normal(0, 0.1)
        reward = np.clip(reward + noise, 0.0, 1.0)

        return reward

    def _calculate_hypothesis_similarity(self, hypothesis: Hypothesis) -> float:
        """Calculate similarity with existing hypotheses"""
        if not self.hypotheses:
            return 0.0

        # Simple word overlap
        words1 = set(hypothesis.statement.lower().split())

        similarities = []
        for h_stats in self.hypotheses:
            words2 = set(h_stats.hypothesis.statement.lower().split())

            if not words1 or not words2:
                similarities.append(0.0)
                continue

            intersection = words1 & words2
            union = words1 | words2

            jaccard = len(intersection) / len(union)
            similarities.append(jaccard)

        return max(similarities)

    def _calculate_paper_support(
        self, hypothesis: Hypothesis, papers: List[Paper]
    ) -> float:
        """Calculate support from papers"""
        if not papers:
            return 0.5  # Neutral

        # Check if hypothesis concepts appear in papers
        hyp_words = set(hypothesis.statement.lower().split())

        support_scores = []

        for paper in papers[:5]:  # Check top 5 papers
            paper_text = f"{paper.title} {paper.abstract}".lower()
            paper_words = set(paper_text.split())

            # Word overlap
            overlap = len(hyp_words & paper_words) / max(len(hyp_words), 1)
            support_scores.append(overlap)

        return np.mean(support_scores) if support_scores else 0.5

    def _update(self, hypothesis: HypothesisWithStats, reward: float) -> None:
        """
        Update hypothesis statistics

        Args:
            hypothesis: Hypothesis to update
            reward: Observed reward
        """
        # Update beta distribution
        if reward > 0.5:
            # Success
            hypothesis.alpha += 1.0
        else:
            # Failure
            hypothesis.beta += 1.0

        # Update samples and mean reward
        hypothesis.samples += 1
        hypothesis.mean_reward = (
            hypothesis.mean_reward * (hypothesis.samples - 1) + reward
        ) / hypothesis.samples

        # Apply decay to old hypotheses
        for h in self.hypotheses:
            if h != hypothesis:
                # Slight decay
                decay = self.decay_factor ** (1.0 / len(self.hypotheses))
                h.alpha *= decay
                h.beta *= decay

    async def _generate_initial_hypotheses(
        self, query: str, papers: List[Paper], n_hypotheses: int
    ) -> List[Hypothesis]:
        """
        Generate initial diverse hypotheses

        Args:
            query: Research query
            papers: Relevant papers
            n_hypotheses: Number of hypotheses to generate

        Returns:
            List of hypotheses
        """
        # Generate diverse hypotheses
        # In production, would use LLM to generate based on papers

        hypotheses = []

        # Strategy 1: Direct hypothesis from query
        hypotheses.append(
            Hypothesis(
                statement=f"Direct hypothesis: {query}",
                rationale="Based on original query",
                testable_predictions=["Prediction 1", "Prediction 2"],
                expected_evidence="Experimental results",
                confidence=0.7,
            )
        )

        # Strategy 2: Contradictory hypothesis
        hypotheses.append(
            Hypothesis(
                statement=f"Alternative hypothesis: Opposite of {query}",
                rationale="Contrarian view",
                testable_predictions=["Prediction A", "Prediction B"],
                expected_evidence="Contradictory evidence",
                confidence=0.5,
            )
        )

        # Strategy 3: Mechanistic hypothesis
        hypotheses.append(
            Hypothesis(
                statement=f"Mechanistic explanation for {query}",
                rationale="Underlying mechanism",
                testable_predictions=["Mechanism test 1", "Mechanism test 2"],
                expected_evidence="Mechanistic evidence",
                confidence=0.6,
            )
        )

        # Strategy 4: Conditional hypothesis
        hypotheses.append(
            Hypothesis(
                statement=f"Conditional on specific factors: {query}",
                rationale="Context-dependent",
                testable_predictions=["Context test 1", "Context test 2"],
                expected_evidence="Conditional evidence",
                confidence=0.6,
            )
        )

        # Strategy 5: Quantitative hypothesis
        hypotheses.append(
            Hypothesis(
                statement=f"Quantitative relationship in {query}",
                rationale="Precise measurement",
                testable_predictions=["Measurement 1", "Measurement 2"],
                expected_evidence="Quantitative data",
                confidence=0.7,
            )
        )

        # Pad with variations if needed
        while len(hypotheses) < n_hypotheses:
            base_hyp = hypotheses[len(hypotheses) % len(hypotheses)]
            variation = Hypothesis(
                statement=f"Variation: {base_hyp.statement}",
                rationale=f"Variant of {base_hyp.rationale}",
                testable_predictions=base_hyp.testable_predictions,
                expected_evidence=base_hyp.expected_evidence,
                confidence=base_hyp.confidence * 0.9,
            )
            hypotheses.append(variation)

        return hypotheses[:n_hypotheses]

    def get_best_hypotheses(self, n: int = 5) -> List[HypothesisWithStats]:
        """
        Get top N hypotheses

        Args:
            n: Number of hypotheses to return

        Returns:
            Top hypotheses
        """
        sorted_hypotheses = sorted(
            self.hypotheses, key=lambda h: h.expected_value, reverse=True
        )
        return sorted_hypotheses[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics"""
        return {
            "total_hypotheses": len(self.hypotheses),
            "total_samples": sum(h.samples for h in self.hypotheses),
            "best_expected_value": max(h.expected_value for h in self.hypotheses)
            if self.hypotheses
            else 0.0,
            "mean_uncertainty": np.mean([h.uncertainty for h in self.hypotheses])
            if self.hypotheses
            else 0.0,
        }
