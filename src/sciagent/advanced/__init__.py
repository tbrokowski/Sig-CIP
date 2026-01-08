"""
Advanced techniques for SciAgent

Includes:
- Bayesian experimental design
- MCTS planning
- Thompson sampling
- Reflexion
- Extended thinking
"""

from sciagent.advanced.bayesian_design import (
    BayesianExperimentSelector,
    ExperimentCandidate,
)
from sciagent.advanced.mcts import MCTSPlanner, MCTSNode, ValueFunctionLearner
from sciagent.advanced.thompson_sampling import (
    ThompsonSamplingExplorer,
    HypothesisWithStats,
)

__all__ = [
    "BayesianExperimentSelector",
    "ExperimentCandidate",
    "MCTSPlanner",
    "MCTSNode",
    "ValueFunctionLearner",
    "ThompsonSamplingExplorer",
    "HypothesisWithStats",
]
