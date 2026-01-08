"""
Python API for SciAgent

High-level Python interface for running experiments programmatically.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from sciagent.coordinator import AdvancedCoordinator
from sciagent.utils.config import Config, load_config
from sciagent.utils.models import ExperimentResult, HumanApprovalRequest


class SciAgent:
    """
    High-level Python API for SciAgent

    Example:
        >>> agent = SciAgent()
        >>> result = await agent.run("Test dropout on CIFAR-10")
        >>> print(result.analysis.summary)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize SciAgent

        Args:
            config: Optional configuration. If None, loads from default location.
        """
        self.config = config or load_config()
        self.coordinator = AdvancedCoordinator(self.config)

    async def run(
        self,
        query: str,
        interactive: bool = True,
        budget: float = 10000,
        on_approval: Optional[Callable] = None,
    ) -> ExperimentResult:
        """
        Run experiment with optional callbacks

        Args:
            query: Research question or task description
            interactive: Enable human-in-the-loop mode
            budget: Compute budget in USD (not enforced yet)
            on_approval: Optional callback for approval requests

        Returns:
            Complete experiment result

        Example:
            >>> agent = SciAgent()
            >>>
            >>> async def handle_approval(request):
            ...     # Custom approval logic
            ...     return True
            >>>
            >>> result = await agent.run(
            ...     "Test dropout on CIFAR-10",
            ...     on_approval=handle_approval
            ... )
        """

        return await self.coordinator.run_experiment(
            query=query,
            user_callback=on_approval,
            auto_approve=not interactive,
        )

    async def resume(self, experiment_id: str) -> ExperimentResult:
        """
        Resume a paused experiment

        Args:
            experiment_id: ID of experiment to resume

        Returns:
            Experiment result
        """
        return await self.coordinator.resume_experiment(experiment_id)

    async def pause(self, experiment_id: str) -> None:
        """
        Pause a running experiment

        Args:
            experiment_id: ID of experiment to pause
        """
        await self.coordinator.pause_experiment(experiment_id)

    def respond(self, request_id: str, response: any) -> None:
        """
        Respond to pending approval request

        Args:
            request_id: ID of approval request
            response: User's response
        """
        self.coordinator.respond_to_request(request_id, response)

    async def get_pending_requests(self) -> List[HumanApprovalRequest]:
        """
        Get all pending approval requests

        Returns:
            List of pending approval requests
        """
        return list(self.coordinator.pending_approvals.values())

    def get_experiment_state(self, experiment_id: str) -> Optional[any]:
        """
        Get current state of an experiment

        Args:
            experiment_id: ID of experiment

        Returns:
            Experiment state or None if not found
        """
        return self.coordinator.experiments.get(experiment_id)

    def list_experiments(self) -> List[str]:
        """
        List all experiment IDs

        Returns:
            List of experiment IDs
        """
        return list(self.coordinator.experiments.keys())
