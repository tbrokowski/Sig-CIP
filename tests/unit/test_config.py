"""
Test configuration management
"""

import pytest
from pathlib import Path
from sciagent.utils.config import Config, AgentConfig


def test_config_creation():
    """Test creating a default configuration"""
    config = Config()

    assert config.science_agent is not None
    assert config.coding_agent is not None
    assert config.overseer_agent is not None
    assert config.cache_dir.exists()
    assert config.data_dir.exists()


def test_agent_config():
    """Test agent configuration"""
    agent_config = AgentConfig(
        enabled=True,
        model="test-model",
        temperature=0.5,
        max_tokens=1000
    )

    assert agent_config.enabled is True
    assert agent_config.model == "test-model"
    assert agent_config.temperature == 0.5
    assert agent_config.max_tokens == 1000


def test_config_paths():
    """Test that config creates necessary directories"""
    config = Config()

    assert config.cache_dir.is_dir()
    assert config.data_dir.is_dir()
    assert config.experiments_dir.is_dir()


def test_config_features():
    """Test feature flags"""
    config = Config()

    assert config.enable_extended_thinking is True
    assert config.enable_reflexion is True
    assert config.enable_debate is True
