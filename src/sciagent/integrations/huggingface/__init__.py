"""
HuggingFace integration for models and datasets
"""

from sciagent.integrations.huggingface.models import HuggingFaceModelManager
from sciagent.integrations.huggingface.datasets import HuggingFaceDatasetManager

__all__ = [
    "HuggingFaceModelManager",
    "HuggingFaceDatasetManager",
]
