"""
Advanced Coordinator with Human-in-the-Loop

Orchestrates all agents with pause/resume, human approval requests,
and process management capabilities.
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from sciagent.utils.config import Config
from sciagent.utils.logging import logger
from sciagent.utils.models import (
    ExperimentResult,
    ExperimentState,
    HumanApprovalRequest,
    ProcessInfo,
    ProcessState,
    Refinement,
)


class UserRejectedError(Exception):
    """Raised when user rejects an action"""
    pass


class AdvancedCoordinator:
    """
    Orchestrates all agents with pause/resume, human-in-the-loop,
    and process management capabilities
    """

    def __init__(self, config: Config):
        self.config = config

        # Lazy load agents to avoid circular imports
        self._agents: Dict[str, Any] = {}

        # Message broker (async queue)
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # State management
        self.experiments: Dict[str, ExperimentState] = {}
        self.processes: Dict[str, ProcessInfo] = {}

        # Human-in-the-loop
        self.pending_approvals: Dict[str, HumanApprovalRequest] = {}
        self.user_callback: Optional[Callable] = None

        # Background task manager
        self.background_tasks: set = set()

    @property
    def agents(self) -> Dict[str, Any]:
        """Lazy load agents"""
        if not self._agents:
            from sciagent.agents.science_agent import ScienceAgent
            from sciagent.agents.coding_agent import CodingAgent
            from sciagent.agents.data_agent import DataAgent
            from sciagent.agents.overseer_agent import OverseerAgent

            self._agents = {
                "science": ScienceAgent(self.config),
                "coding": CodingAgent(self.config),
                "data": DataAgent(self.config),
                "overseer": OverseerAgent(self.config),
            }
        return self._agents

    async def run_experiment(
        self,
        query: str,
        user_callback: Optional[Callable] = None,
        auto_approve: bool = False,
    ) -> ExperimentResult:
        """
        Main experiment workflow with human-in-the-loop

        Args:
            query: Research question or task description
            user_callback: Callback for user interaction
            auto_approve: Skip approval requests (automated mode)

        Returns:
            Complete experiment result
        """

        self.user_callback = user_callback
        experiment_id = self._generate_experiment_id()

        # Initialize experiment state
        state = ExperimentState(
            experiment_id=experiment_id,
            query=query,
            status=ProcessState.RUNNING,
            checkpoints=[],
        )
        self.experiments[experiment_id] = state

        try:
            # Phase 1: Scientific Planning
            logger.info(f"[{experiment_id}] Phase 1: Scientific Planning")

            planning = await self._run_with_approval(
                agent_name="science",
                action="plan_experiments",
                payload={"query": query},
                approval_message="Science agent has proposed experiment designs. Review?",
                auto_approve=auto_approve,
            )

            state.add_checkpoint("planning_complete", planning)

            # Phase 2: Data Preparation
            logger.info(f"[{experiment_id}] Phase 2: Data Preparation")

            data_prep = None
            if planning.requires_dataset:
                data_prep = await self._run_with_approval(
                    agent_name="data",
                    action="prepare_data",
                    payload={"dataset_info": planning.dataset_requirements},
                    approval_message=f"Ready to download {planning.dataset_requirements.name}?",
                    auto_approve=auto_approve,
                )
                state.add_checkpoint("data_prepared", data_prep)

            # Phase 3: Code Generation
            logger.info(f"[{experiment_id}] Phase 3: Code Generation")

            code = await self._run_with_approval(
                agent_name="coding",
                action="implement_experiment",
                payload={
                    "plan": planning,
                    "data_info": data_prep,
                },
                approval_message="Review generated code before execution?",
                auto_approve=auto_approve,
            )

            state.add_checkpoint("code_generated", code)

            # Phase 4: Execution
            logger.info(f"[{experiment_id}] Phase 4: Execution")

            execution = await self._execute_with_monitoring(
                experiment_id=experiment_id,
                code=code,
                user_callback=user_callback,
            )

            state.add_checkpoint("execution_complete", execution)

            # Phase 5: Analysis & Validation
            logger.info(f"[{experiment_id}] Phase 5: Analysis")

            # Multi-agent analysis
            analyses = await asyncio.gather(
                # Science agent analyzes results
                self.agents["science"].analyze_results(
                    planning.recommendation.hypothesis, execution
                ),
                # Overseer validates
                self.agents["overseer"].validate_experiment(
                    code, execution, planning.recommendation.hypothesis
                ),
            )

            science_analysis, validation = analyses
            state.add_checkpoint("analysis_complete", science_analysis)

            # Phase 6: Refinement Proposal
            logger.info(f"[{experiment_id}] Phase 6: Refinement Proposal")

            refinements = await self.agents["science"].propose_refinements(
                query=query,
                planning=planning,
                results=execution,
                analysis=science_analysis,
            )

            # Ask user if they want to refine
            if refinements and not auto_approve:
                decision = await self._request_user_decision(
                    message=f"Analysis complete. {science_analysis.summary}\n\nProposed refinements:",
                    options=[f"({i+1}) {r.description}" for i, r in enumerate(refinements)]
                    + ["(0) Complete experiment"],
                    allow_multiple=True,
                )

                if decision and decision != ["0"]:
                    # User wants refinements - spawn sub-experiments
                    for refinement_choice in decision:
                        idx = int(refinement_choice) - 1
                        if 0 <= idx < len(refinements):
                            await self._spawn_refinement_experiment(
                                parent_id=experiment_id,
                                refinement=refinements[idx],
                                user_callback=user_callback,
                            )

            state.status = ProcessState.COMPLETED

            return ExperimentResult(
                experiment_id=experiment_id,
                query=query,
                planning=planning,
                code=code,
                execution=execution,
                analysis=science_analysis,
                validation=validation,
                refinements=refinements,
                state=state,
            )

        except Exception as e:
            state.status = ProcessState.FAILED
            state.error = str(e)

            # Ask user how to handle error
            if user_callback:
                recovery = await self._request_user_decision(
                    message=f"Experiment failed: {e}\n\nHow to proceed?",
                    options=[
                        "Retry from last checkpoint",
                        "Debug with agent",
                        "Abort experiment",
                    ],
                )

                if recovery == "Retry from last checkpoint":
                    return await self._retry_from_checkpoint(experiment_id)
                elif recovery == "Debug with agent":
                    return await self._debug_with_agent(experiment_id, e)

            raise

    async def _run_with_approval(
        self,
        agent_name: str,
        action: str,
        payload: Dict[str, Any],
        approval_message: str,
        auto_approve: bool = False,
    ) -> Any:
        """
        Run agent action with optional user approval

        Args:
            agent_name: Name of agent to run
            action: Action to perform
            payload: Action parameters
            approval_message: Message to show for approval
            auto_approve: Skip approval

        Returns:
            Result from agent
        """

        # Pause for user approval if needed
        if not auto_approve and self.user_callback:
            approved = await self._request_user_approval(
                message=approval_message,
                context={"agent": agent_name, "action": action},
            )

            if not approved:
                raise UserRejectedError(f"User rejected {action}")

        # Run agent action
        agent = self.agents[agent_name]
        result = await agent.process({"action": action, **payload})

        return result

    async def _execute_with_monitoring(
        self,
        experiment_id: str,
        code: Any,
        user_callback: Optional[Callable] = None,
    ) -> Any:
        """
        Execute code with real-time monitoring and error handling

        Args:
            experiment_id: ID of experiment
            code: Code artifact to execute
            user_callback: Optional user callback

        Returns:
            Execution result
        """
        from sciagent.execution.executor import MonitoredExecutor

        # Create monitored execution environment
        executor = MonitoredExecutor(
            code=code,
            experiment_id=experiment_id,
            on_error=lambda e: self._handle_execution_error(experiment_id, e),
            on_progress=lambda p: self._report_progress(experiment_id, p),
        )

        # Run in background
        task = asyncio.create_task(executor.run())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        # Wait for completion or error
        result = await task

        return result

    async def _handle_execution_error(self, experiment_id: str, error: Exception):
        """
        Handle runtime errors by pausing and asking user

        Args:
            experiment_id: ID of experiment
            error: Exception that occurred
        """

        state = self.experiments[experiment_id]
        state.status = ProcessState.PAUSED

        if self.user_callback:
            decision = await self._request_user_decision(
                message=f"Execution error: {error}\n\nHow to proceed?",
                options=[
                    "Let agent debug and retry",
                    "Show me the error details",
                    "Abort execution",
                ],
            )

            if decision == "Let agent debug and retry":
                # Send to coding agent for debugging
                fixed_code = await self.agents["coding"].debug_code(
                    code=state.checkpoints[-1].data,  # Get last code checkpoint
                    error=str(error),
                    traceback=traceback.format_exc(),
                )

                # Retry with fixed code
                state.status = ProcessState.RUNNING
                return await self._execute_with_monitoring(
                    experiment_id, fixed_code, self.user_callback
                )

    async def _report_progress(self, experiment_id: str, progress: Dict[str, Any]):
        """Report progress to user"""
        logger.info(f"[{experiment_id}] Progress: {progress}")

    async def _request_user_approval(
        self,
        message: str,
        context: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Request yes/no approval from user

        Args:
            message: Approval message
            context: Additional context
            timeout: Optional timeout in seconds

        Returns:
            True if approved, False otherwise
        """

        request = HumanApprovalRequest(
            request_id=self._generate_request_id(),
            request_type="approval",
            context=context,
            options=["Yes", "No"],
            timeout=timeout,
        )

        self.pending_approvals[request.request_id] = request

        # Call user callback
        if self.user_callback:
            await self.user_callback(request)

        # Wait for response
        response = await request.response

        return response.lower() in ["yes", "y", "true", "1"]

    async def _request_user_decision(
        self,
        message: str,
        options: List[str],
        allow_multiple: bool = False,
    ) -> Union[str, List[str]]:
        """
        Request user to choose from options

        Args:
            message: Decision message
            options: List of options
            allow_multiple: Allow multiple selections

        Returns:
            Selected option(s)
        """

        request = HumanApprovalRequest(
            request_id=self._generate_request_id(),
            request_type="choice",
            context={"message": message, "allow_multiple": allow_multiple},
            options=options,
        )

        self.pending_approvals[request.request_id] = request

        if self.user_callback:
            await self.user_callback(request)

        response = await request.response

        return response

    def respond_to_request(self, request_id: str, response: Any):
        """
        User responds to pending approval request

        Args:
            request_id: ID of request
            response: User's response
        """

        if request_id in self.pending_approvals:
            request = self.pending_approvals[request_id]
            request.response.set_result(response)
            del self.pending_approvals[request_id]
        else:
            raise ValueError(f"No pending request with id {request_id}")

    async def _spawn_refinement_experiment(
        self,
        parent_id: str,
        refinement: Refinement,
        user_callback: Optional[Callable],
    ):
        """
        Spawn a new experiment for refinement

        Args:
            parent_id: Parent experiment ID
            refinement: Refinement to explore
            user_callback: User callback
        """

        # Create refinement as sub-experiment
        refinement_query = (
            f"{self.experiments[parent_id].query} - Refinement: {refinement.description}"
        )

        # Run as background task
        task = asyncio.create_task(
            self.run_experiment(
                query=refinement_query,
                user_callback=user_callback,
                auto_approve=False,
            )
        )

        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        # Link to parent
        self.experiments[parent_id].refinement_experiments.append(task)

        logger.info(f"Spawned refinement experiment for {parent_id}")

    async def pause_experiment(self, experiment_id: str):
        """Pause a running experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = ProcessState.PAUSED

    async def resume_experiment(self, experiment_id: str) -> ExperimentResult:
        """Resume a paused experiment"""
        if experiment_id in self.experiments:
            state = self.experiments[experiment_id]
            if state.status == ProcessState.PAUSED:
                state.status = ProcessState.RUNNING
                # Resume from last checkpoint
                return await self._retry_from_checkpoint(experiment_id)
        raise ValueError(f"No paused experiment with id {experiment_id}")

    async def _retry_from_checkpoint(self, experiment_id: str) -> ExperimentResult:
        """Retry experiment from last checkpoint"""
        # TODO: Implement checkpoint recovery
        logger.warning("Checkpoint recovery not yet implemented")
        raise NotImplementedError("Checkpoint recovery not yet implemented")

    async def _debug_with_agent(
        self, experiment_id: str, error: Exception
    ) -> ExperimentResult:
        """Debug experiment with agent assistance"""
        # TODO: Implement agent-assisted debugging
        logger.warning("Agent-assisted debugging not yet implemented")
        raise NotImplementedError("Agent-assisted debugging not yet implemented")

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        return f"exp_{uuid.uuid4().hex[:8]}"

    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"req_{uuid.uuid4().hex[:8]}"
