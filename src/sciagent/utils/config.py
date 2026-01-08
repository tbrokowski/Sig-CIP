"""
Configuration management for SciAgent
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    enabled: bool = True
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 8000
    timeout: int = 300


@dataclass
class MCPServerConfig:
    """Configuration for MCP server"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for SciAgent"""

    # API Keys (from environment)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Agent configurations
    science_agent: AgentConfig = field(default_factory=lambda: AgentConfig(
        model="gemini-2.0-flash-thinking-exp"
    ))
    coding_agent: AgentConfig = field(default_factory=lambda: AgentConfig(
        model="claude-sonnet-4-20250514"
    ))
    overseer_agent: AgentConfig = field(default_factory=lambda: AgentConfig(
        model="gpt-4"
    ))

    # MCP servers
    mcp_servers: List[MCPServerConfig] = field(default_factory=list)

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".sciagent" / "cache")
    data_dir: Path = field(default_factory=lambda: Path.home() / ".sciagent" / "datasets")
    experiments_dir: Path = field(default_factory=lambda: Path.home() / ".sciagent" / "experiments")

    # Execution settings
    max_concurrent_experiments: int = 5
    default_timeout: int = 3600
    enable_sandbox: bool = True

    # Advanced features
    enable_extended_thinking: bool = True
    enable_reflexion: bool = True
    enable_debate: bool = True
    enable_mcts: bool = True
    enable_bayesian_design: bool = True

    # Reflexion settings
    reflexion_max_iterations: int = 5
    reflexion_quality_threshold: float = 0.95

    # MCTS settings
    mcts_simulations: int = 100
    mcts_exploration_constant: float = 1.414

    # Debate settings
    debate_rounds: int = 3
    debate_consensus_threshold: float = 0.75

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def __post_init__(self):
        """Create directories if they don't exist"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load configuration from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse agent configs
        if "agents" in data:
            for agent_name, agent_data in data["agents"].items():
                if hasattr(cls, f"{agent_name}_agent"):
                    agent_config = AgentConfig(**agent_data)
                    data[f"{agent_name}_agent"] = agent_config

        # Parse MCP servers
        if "mcp_servers" in data:
            mcp_servers = [
                MCPServerConfig(**server_data)
                for server_data in data["mcp_servers"]
            ]
            data["mcp_servers"] = mcp_servers

        # Convert path strings to Path objects
        for key in ["cache_dir", "data_dir", "experiments_dir", "log_file"]:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file"""
        data = {
            "agents": {
                "science": {
                    "enabled": self.science_agent.enabled,
                    "model": self.science_agent.model,
                    "temperature": self.science_agent.temperature,
                    "max_tokens": self.science_agent.max_tokens,
                },
                "coding": {
                    "enabled": self.coding_agent.enabled,
                    "model": self.coding_agent.model,
                    "temperature": self.coding_agent.temperature,
                    "max_tokens": self.coding_agent.max_tokens,
                },
                "overseer": {
                    "enabled": self.overseer_agent.enabled,
                    "model": self.overseer_agent.model,
                    "temperature": self.overseer_agent.temperature,
                    "max_tokens": self.overseer_agent.max_tokens,
                },
            },
            "mcp_servers": [
                {
                    "name": server.name,
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                }
                for server in self.mcp_servers
            ],
            "paths": {
                "cache_dir": str(self.cache_dir),
                "data_dir": str(self.data_dir),
                "experiments_dir": str(self.experiments_dir),
            },
            "execution": {
                "max_concurrent_experiments": self.max_concurrent_experiments,
                "default_timeout": self.default_timeout,
                "enable_sandbox": self.enable_sandbox,
            },
            "features": {
                "enable_extended_thinking": self.enable_extended_thinking,
                "enable_reflexion": self.enable_reflexion,
                "enable_debate": self.enable_debate,
                "enable_mcts": self.enable_mcts,
                "enable_bayesian_design": self.enable_bayesian_design,
            },
            "logging": {
                "log_level": self.log_level,
                "log_file": str(self.log_file) if self.log_file else None,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or create default"""
    if config_path is None:
        config_path = Path.home() / ".sciagent" / "config.yaml"

    if config_path.exists():
        return Config.from_yaml(config_path)
    else:
        # Create default config
        config = Config()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(config_path)
        return config
