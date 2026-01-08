"""
Monte Carlo Tree Search (MCTS) for Experiment Planning

Uses MCTS to plan optimal sequences of experiments to test hypotheses.
Each node represents a state of knowledge, and edges represent experiments.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from sciagent.utils.logging import logger
from sciagent.utils.models import ExperimentDesign, ExperimentSequence, Hypothesis


@dataclass
class MCTSNode:
    """Node in MCTS tree"""

    state: Dict[str, Any]
    parent: Optional[MCTSNode] = None
    children: List[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[ExperimentDesign] = field(default_factory=list)
    action: Optional[ExperimentDesign] = None  # Action that led to this node

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound for Trees (UCT)"""
        if self.visits == 0:
            return float("inf")

        exploitation = self.value / self.visits

        if self.parent is None or self.parent.visits == 0:
            exploration = 0.0
        else:
            exploration = math.sqrt(
                2 * math.log(self.parent.visits) / self.visits
            )

        return exploitation + exploration


class MCTSPlanner:
    """
    MCTS-based experiment planner

    Plans optimal sequences of experiments to maximize expected value
    while minimizing cost.
    """

    def __init__(
        self,
        exploration_constant: float = 1.414,
        n_simulations: int = 100,
        max_depth: int = 5,
        value_function: Optional[Callable] = None,
    ):
        """
        Initialize MCTS planner

        Args:
            exploration_constant: UCT exploration constant (sqrt(2) by default)
            n_simulations: Number of MCTS simulations
            max_depth: Maximum tree depth
            value_function: Custom value function for states
        """
        self.exploration_constant = exploration_constant
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.value_function = value_function or self._default_value_function

    async def plan(
        self,
        hypothesis: Hypothesis,
        initial_design: ExperimentDesign,
        budget: float = 10000,
    ) -> ExperimentSequence:
        """
        Plan optimal experiment sequence using MCTS

        Args:
            hypothesis: Hypothesis to test
            initial_design: Initial experiment design
            budget: Available budget

        Returns:
            Planned experiment sequence
        """
        logger.info(
            f"Planning experiment sequence with MCTS ({self.n_simulations} simulations)"
        )

        # Initialize root node
        initial_state = {
            "hypothesis": hypothesis,
            "knowledge": {},
            "budget_remaining": budget,
            "depth": 0,
        }

        root = MCTSNode(
            state=initial_state,
            untried_actions=self._generate_possible_experiments(
                hypothesis, initial_design
            ),
        )

        # Run MCTS simulations
        for i in range(self.n_simulations):
            node = root

            # Selection
            while node.untried_actions == [] and node.children:
                node = self._select_child(node)

            # Expansion
            if node.untried_actions and node.state["depth"] < self.max_depth:
                action = random.choice(node.untried_actions)
                node = self._expand(node, action)

            # Simulation
            value = await self._simulate(node)

            # Backpropagation
            self._backpropagate(node, value)

            if (i + 1) % 20 == 0:
                logger.debug(
                    f"MCTS simulation {i+1}/{self.n_simulations}, "
                    f"root value: {root.value/max(root.visits, 1):.3f}"
                )

        # Extract best sequence
        sequence = self._extract_best_sequence(root)

        logger.info(
            f"Planned sequence of {len(sequence.steps)} experiments, "
            f"expected value: {sequence.total_value:.3f}"
        )

        return sequence

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child node using UCT

        Args:
            node: Current node

        Returns:
            Selected child node
        """
        # Select child with highest UCB1 value
        return max(node.children, key=lambda c: c.ucb1)

    def _expand(self, node: MCTSNode, action: ExperimentDesign) -> MCTSNode:
        """
        Expand node with new action

        Args:
            node: Node to expand
            action: Action to take

        Returns:
            New child node
        """
        # Create new state by applying action
        new_state = {
            "hypothesis": node.state["hypothesis"],
            "knowledge": {**node.state["knowledge"], action.description: "planned"},
            "budget_remaining": node.state["budget_remaining"] - 100,  # Simplified cost
            "depth": node.state["depth"] + 1,
        }

        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            untried_actions=self._generate_possible_experiments(
                node.state["hypothesis"], action
            ),
        )

        # Remove action from untried actions and add child
        node.untried_actions.remove(action)
        node.children.append(child)

        return child

    async def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate from node to terminal state

        Args:
            node: Starting node

        Returns:
            Simulated value
        """
        # Simple random rollout
        current_state = node.state.copy()
        total_value = 0.0
        depth = current_state["depth"]

        # Rollout until max depth or budget exhausted
        while depth < self.max_depth and current_state["budget_remaining"] > 0:
            # Random action
            value = self.value_function(current_state)
            total_value += value * (0.9 ** depth)  # Discount factor

            # Update state
            current_state["budget_remaining"] -= 100
            depth += 1

        return total_value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate value up the tree

        Args:
            node: Starting node
            value: Value to propagate
        """
        current = node

        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def _extract_best_sequence(self, root: MCTSNode) -> ExperimentSequence:
        """
        Extract best experiment sequence from tree

        Args:
            root: Root node

        Returns:
            Best experiment sequence
        """
        sequence = []
        total_value = 0.0
        total_cost = 0.0
        current = root

        # Follow most visited path
        while current.children:
            current = max(current.children, key=lambda c: c.visits)
            if current.action:
                sequence.append(current.action)
                total_value += current.value / max(current.visits, 1)
                total_cost += 100  # Simplified cost

        return ExperimentSequence(
            steps=sequence,
            total_value=total_value,
            expected_cost=total_cost,
            expected_duration=len(sequence) * 3600,  # 1 hour per experiment
        )

    def _generate_possible_experiments(
        self, hypothesis: Hypothesis, base_design: ExperimentDesign
    ) -> List[ExperimentDesign]:
        """
        Generate possible next experiments

        Args:
            hypothesis: Current hypothesis
            base_design: Base experiment design

        Returns:
            List of possible experiments
        """
        # Generate variations of base design
        experiments = []

        # Variation 1: Increase sample size
        exp1 = ExperimentDesign(
            description=f"{base_design.description} (larger sample)",
            measurements=base_design.measurements,
            controls=base_design.controls,
            sample_size=base_design.sample_size * 2,
            statistical_tests=base_design.statistical_tests,
            potential_confounds=base_design.potential_confounds,
            success_criteria=base_design.success_criteria,
        )
        experiments.append(exp1)

        # Variation 2: Add more controls
        exp2 = ExperimentDesign(
            description=f"{base_design.description} (more controls)",
            measurements=base_design.measurements,
            controls=base_design.controls + ["additional_control"],
            sample_size=base_design.sample_size,
            statistical_tests=base_design.statistical_tests,
            potential_confounds=base_design.potential_confounds,
            success_criteria=base_design.success_criteria,
        )
        experiments.append(exp2)

        # Variation 3: Different statistical tests
        exp3 = ExperimentDesign(
            description=f"{base_design.description} (robust tests)",
            measurements=base_design.measurements,
            controls=base_design.controls,
            sample_size=base_design.sample_size,
            statistical_tests=["bootstrap", "permutation_test"],
            potential_confounds=base_design.potential_confounds,
            success_criteria=base_design.success_criteria,
        )
        experiments.append(exp3)

        return experiments

    def _default_value_function(self, state: Dict[str, Any]) -> float:
        """
        Default value function for states

        Args:
            state: Current state

        Returns:
            Estimated value
        """
        # Simple heuristic: value decreases with depth, increases with knowledge
        depth_penalty = 1.0 / (1.0 + state["depth"])
        knowledge_bonus = len(state["knowledge"]) * 0.1
        budget_factor = state["budget_remaining"] / 10000

        value = depth_penalty + knowledge_bonus + budget_factor

        return value

    def visualize_tree(self, root: MCTSNode, max_depth: int = 3) -> str:
        """
        Visualize MCTS tree (text representation)

        Args:
            root: Root node
            max_depth: Maximum depth to visualize

        Returns:
            Text representation of tree
        """
        lines = []

        def _traverse(node: MCTSNode, depth: int, prefix: str):
            if depth > max_depth:
                return

            indent = "  " * depth
            action_desc = node.action.description if node.action else "Root"
            value = node.value / max(node.visits, 1)

            lines.append(
                f"{indent}{prefix}{action_desc} "
                f"(visits: {node.visits}, value: {value:.3f})"
            )

            for i, child in enumerate(node.children):
                is_last = i == len(node.children) - 1
                child_prefix = "└─ " if is_last else "├─ "
                _traverse(child, depth + 1, child_prefix)

        _traverse(root, 0, "")

        return "\n".join(lines)


class ValueFunctionLearner:
    """
    Learn value function from experiment history

    Uses past experiments to learn which states are valuable.
    """

    def __init__(self):
        """Initialize value function learner"""
        self.history: List[Tuple[Dict[str, Any], float]] = []
        self.model: Optional[Any] = None

    def add_experience(self, state: Dict[str, Any], value: float) -> None:
        """
        Add experience to history

        Args:
            state: State
            value: Observed value
        """
        self.history.append((state, value))

    def train(self) -> None:
        """Train value function on history"""
        if len(self.history) < 10:
            logger.warning("Not enough data to train value function")
            return

        # In production, would train ML model here
        # For now, use simple average
        logger.info(f"Trained value function on {len(self.history)} examples")

    def predict(self, state: Dict[str, Any]) -> float:
        """
        Predict value for state

        Args:
            state: State to evaluate

        Returns:
            Predicted value
        """
        if not self.history:
            return 0.5  # Default

        # Simple: return average of recent values
        recent_values = [v for _, v in self.history[-10:]]
        return np.mean(recent_values)
