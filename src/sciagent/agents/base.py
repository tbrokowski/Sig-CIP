"""
Base agent class
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from sciagent.utils.config import Config
from sciagent.utils.logging import logger


class BaseAgent(ABC):
    """
    Base class for all agents in SciAgent system

    All agents must implement the process() method which handles
    incoming requests and returns results.
    """

    def __init__(self, config: Config):
        """
        Initialize agent

        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Any:
        """
        Process a request and return result

        Args:
            request: Request dictionary with 'action' key and other parameters

        Returns:
            Result of processing
        """
        pass

    async def initialize(self) -> None:
        """
        Optional initialization hook called before first use

        Override this to perform any setup needed
        """
        pass

    async def cleanup(self) -> None:
        """
        Optional cleanup hook called when shutting down

        Override this to perform any cleanup needed
        """
        pass

    def _validate_request(self, request: Dict[str, Any], required_keys: list[str]) -> None:
        """
        Validate that request contains required keys

        Args:
            request: Request to validate
            required_keys: List of required keys

        Raises:
            ValueError: If required keys are missing
        """
        missing = [key for key in required_keys if key not in request]
        if missing:
            raise ValueError(f"Missing required keys in request: {missing}")
