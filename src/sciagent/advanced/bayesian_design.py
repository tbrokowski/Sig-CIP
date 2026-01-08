"""
Bayesian Experimental Design

Uses Bayesian optimization to select the most informative next experiments.
Implements Expected Information Gain (EIG) to guide experiment selection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from sciagent.utils.logging import logger


@dataclass
class ExperimentCandidate:
    """A candidate experiment to be evaluated"""

    description: str
    parameters: Dict[str, Any]
    expected_information_gain: float = 0.0
    expected_cost: float = 1.0
    uncertainty: float = 1.0
    information_gain_explanation: str = ""
    estimated_cost: float = 0.0


class BayesianExperimentSelector:
    """
    Bayesian experiment selector using Expected Information Gain

    Selects experiments that are expected to provide the most information
    about hypotheses while considering cost and feasibility.
    """

    def __init__(
        self,
        exploration_weight: float = 1.0,
        cost_weight: float = 0.3,
        use_thompson_sampling: bool = True,
    ):
        """
        Initialize Bayesian experiment selector

        Args:
            exploration_weight: Weight for exploration vs exploitation
            cost_weight: Weight for cost in selection
            use_thompson_sampling: Use Thompson sampling for selection
        """
        self.exploration_weight = exploration_weight
        self.cost_weight = cost_weight
        self.use_thompson_sampling = use_thompson_sampling

        # Prior beliefs (beta distribution parameters)
        self.prior_alpha = 1.0
        self.prior_beta = 1.0

        # History of experiments and results
        self.experiment_history: List[Tuple[ExperimentCandidate, Any]] = []

    async def select_experiments(
        self,
        candidates: List[ExperimentCandidate],
        prior_results: List[Any],
        max_select: int = 3,
    ) -> List[ExperimentCandidate]:
        """
        Select most informative experiments using Bayesian optimization

        Args:
            candidates: List of candidate experiments
            prior_results: Results from previous experiments
            max_select: Maximum number of experiments to select

        Returns:
            Selected experiments ranked by expected information gain
        """
        logger.info(f"Selecting from {len(candidates)} candidate experiments")

        # Calculate expected information gain for each candidate
        for candidate in candidates:
            eig = await self._calculate_expected_information_gain(
                candidate, prior_results
            )

            # Adjust for cost
            cost_adjusted_eig = eig - (self.cost_weight * candidate.expected_cost)

            candidate.expected_information_gain = cost_adjusted_eig

        # Sort by expected information gain
        sorted_candidates = sorted(
            candidates, key=lambda c: c.expected_information_gain, reverse=True
        )

        # Select top experiments
        if self.use_thompson_sampling:
            selected = self._thompson_sample(sorted_candidates, max_select)
        else:
            selected = sorted_candidates[:max_select]

        logger.info(
            f"Selected {len(selected)} experiments with avg EIG: "
            f"{np.mean([c.expected_information_gain for c in selected]):.3f}"
        )

        return selected

    async def _calculate_expected_information_gain(
        self, candidate: ExperimentCandidate, prior_results: List[Any]
    ) -> float:
        """
        Calculate expected information gain for a candidate experiment

        Uses KL divergence between prior and posterior distributions.

        Args:
            candidate: Candidate experiment
            prior_results: Results from previous experiments

        Returns:
            Expected information gain
        """
        # Simple heuristic-based EIG calculation
        # In production, this would use more sophisticated Bayesian methods

        # Base information gain from uncertainty reduction
        base_ig = candidate.uncertainty

        # Bonus for exploring new parameter regions
        novelty_bonus = self._calculate_novelty(candidate)

        # Penalty for redundancy with prior experiments
        redundancy_penalty = self._calculate_redundancy(candidate, prior_results)

        # Expected information gain
        eig = base_ig + (0.3 * novelty_bonus) - (0.2 * redundancy_penalty)

        # Add explanation
        candidate.information_gain_explanation = (
            f"Base IG: {base_ig:.3f}, Novelty: {novelty_bonus:.3f}, "
            f"Redundancy: {redundancy_penalty:.3f}"
        )

        return max(0.0, eig)

    def _calculate_novelty(self, candidate: ExperimentCandidate) -> float:
        """
        Calculate novelty score for a candidate

        Args:
            candidate: Candidate experiment

        Returns:
            Novelty score (0-1)
        """
        if not self.experiment_history:
            return 1.0  # First experiment is maximally novel

        # Compare with historical experiments
        similarities = []

        for past_exp, _ in self.experiment_history:
            similarity = self._parameter_similarity(
                candidate.parameters, past_exp.parameters
            )
            similarities.append(similarity)

        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity

        return novelty

    def _calculate_redundancy(
        self, candidate: ExperimentCandidate, prior_results: List[Any]
    ) -> float:
        """
        Calculate redundancy with prior experiments

        Args:
            candidate: Candidate experiment
            prior_results: Prior experimental results

        Returns:
            Redundancy score (0-1)
        """
        if not prior_results:
            return 0.0

        # Simple redundancy based on parameter overlap
        # In production, would consider result distributions

        redundancy_scores = []

        for past_exp, _ in self.experiment_history[-5:]:  # Look at recent 5
            similarity = self._parameter_similarity(
                candidate.parameters, past_exp.parameters
            )
            redundancy_scores.append(similarity)

        return np.mean(redundancy_scores) if redundancy_scores else 0.0

    def _parameter_similarity(
        self, params1: Dict[str, Any], params2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between parameter sets

        Args:
            params1: First parameter set
            params2: Second parameter set

        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity
        keys1 = set(params1.keys())
        keys2 = set(params2.keys())

        if not keys1 or not keys2:
            return 0.0

        intersection = keys1 & keys2
        union = keys1 | keys2

        jaccard = len(intersection) / len(union)

        # Also consider value similarity for common keys
        value_similarity = 0.0
        if intersection:
            similarities = []
            for key in intersection:
                v1, v2 = params1[key], params2[key]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    # Numeric similarity
                    max_val = max(abs(v1), abs(v2), 1e-8)
                    sim = 1.0 - abs(v1 - v2) / max_val
                    similarities.append(max(0.0, sim))
                elif v1 == v2:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)

            value_similarity = np.mean(similarities)

        # Combine Jaccard and value similarity
        overall_similarity = 0.5 * jaccard + 0.5 * value_similarity

        return overall_similarity

    def _thompson_sample(
        self, candidates: List[ExperimentCandidate], n_samples: int
    ) -> List[ExperimentCandidate]:
        """
        Use Thompson sampling to select experiments

        Args:
            candidates: Candidate experiments
            n_samples: Number of samples to select

        Returns:
            Selected experiments
        """
        # Sample from beta distributions for each candidate
        samples = []

        for candidate in candidates:
            # Update beta distribution based on EIG
            alpha = self.prior_alpha + candidate.expected_information_gain
            beta = self.prior_beta + (1.0 - candidate.expected_information_gain)

            # Sample from beta distribution
            sample = np.random.beta(alpha, beta)
            samples.append((sample, candidate))

        # Sort by sampled values and select top N
        samples.sort(reverse=True, key=lambda x: x[0])
        selected = [candidate for _, candidate in samples[:n_samples]]

        return selected

    def update_with_result(
        self, experiment: ExperimentCandidate, result: Any
    ) -> None:
        """
        Update beliefs based on experiment result

        Args:
            experiment: Conducted experiment
            result: Experimental result
        """
        self.experiment_history.append((experiment, result))

        # Update prior beliefs
        # This is a simplified update; production would use full Bayesian update
        if hasattr(result, "success") and result.success:
            self.prior_alpha += 1.0
        else:
            self.prior_beta += 0.5

        logger.debug(
            f"Updated Bayesian beliefs: α={self.prior_alpha:.2f}, β={self.prior_beta:.2f}"
        )

    def get_posterior_distribution(self) -> Tuple[float, float]:
        """
        Get current posterior distribution parameters

        Returns:
            Tuple of (alpha, beta) for beta distribution
        """
        return (self.prior_alpha, self.prior_beta)

    def reset(self) -> None:
        """Reset to prior beliefs"""
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
        self.experiment_history.clear()
        logger.info("Reset Bayesian experiment selector to prior")


def calculate_mutual_information(
    prior_dist: np.ndarray, posterior_dist: np.ndarray
) -> float:
    """
    Calculate mutual information between prior and posterior

    Args:
        prior_dist: Prior probability distribution
        posterior_dist: Posterior probability distribution

    Returns:
        Mutual information (in nats)
    """
    # Ensure valid probability distributions
    prior_dist = prior_dist / np.sum(prior_dist)
    posterior_dist = posterior_dist / np.sum(posterior_dist)

    # KL divergence
    epsilon = 1e-10  # Avoid log(0)
    kl_div = np.sum(
        posterior_dist * np.log((posterior_dist + epsilon) / (prior_dist + epsilon))
    )

    return kl_div


def expected_information_gain(
    experiment_outcomes: List[float],
    outcome_probabilities: List[float],
    current_belief: float,
) -> float:
    """
    Calculate expected information gain for an experiment

    Args:
        experiment_outcomes: Possible outcomes
        outcome_probabilities: Probabilities of outcomes
        current_belief: Current belief parameter

    Returns:
        Expected information gain
    """
    # Simplified EIG calculation
    # In production, would integrate over outcome space

    eig = 0.0

    for outcome, prob in zip(experiment_outcomes, outcome_probabilities):
        # Expected change in belief
        belief_change = abs(outcome - current_belief)
        eig += prob * belief_change

    return eig
