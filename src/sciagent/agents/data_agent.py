"""
Data Agent - Automated Dataset Management

Handles dataset downloading, preprocessing, and data loader creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sciagent.agents.base import BaseAgent
from sciagent.utils.config import Config
from sciagent.utils.models import DataPreparation, DatasetRequirements


class DataAgent(BaseAgent):
    """
    Autonomous data preparation agent
    Handles dataset downloading, preprocessing, loader creation
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.cache_dir = config.data_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Dataset handlers registry
        self.known_datasets = {
            "cifar10": self._handle_cifar10,
            "cifar-10": self._handle_cifar10,
            "cifar100": self._handle_cifar100,
            "cifar-100": self._handle_cifar100,
            "mnist": self._handle_mnist,
            "imagenet": self._handle_imagenet,
            "coco": self._handle_coco,
        }

    async def process(self, request: Dict[str, Any]) -> Any:
        """
        Process a request

        Supported actions:
        - prepare_data: Download and prepare dataset
        """

        action = request.get("action")

        if action == "prepare_data":
            return await self.prepare_data(request["dataset_info"])
        else:
            raise ValueError(f"Unknown action: {action}")

    async def prepare_data(self, dataset_info: DatasetRequirements) -> DataPreparation:
        """
        Automatically download and prepare dataset

        Args:
            dataset_info: Dataset requirements

        Returns:
            Data preparation result with loaders and code
        """

        dataset_name = dataset_info.name.lower()

        self.logger.info(f"Preparing dataset: {dataset_name}")

        # Check if we have a handler
        if dataset_name in self.known_datasets:
            handler = self.known_datasets[dataset_name]
            return await handler(dataset_info)
        else:
            # Try to auto-discover
            return await self._discover_and_prepare(dataset_info)

    async def _handle_cifar10(self, dataset_info: DatasetRequirements) -> DataPreparation:
        """Handle CIFAR-10 dataset"""

        dataset_name = "CIFAR10"
        cache_path = self.cache_dir / "cifar10"

        # Generate loader code
        loader_code = f"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root='{cache_path}',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='{cache_path}',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=True,
    num_workers={dataset_info.num_workers}
)

test_loader = DataLoader(
    test_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=False,
    num_workers={dataset_info.num_workers}
)

# Dataset info
num_classes = 10
input_shape = (3, 32, 32)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
"""

        return DataPreparation(
            dataset_name=dataset_name,
            cache_path=cache_path,
            loaders={},  # Loaders created in code
            loader_code=loader_code,
            statistics={
                "num_classes": 10,
                "train_size": 50000,
                "test_size": 10000,
                "input_shape": (3, 32, 32),
            },
            samples=[],
        )

    async def _handle_cifar100(self, dataset_info: DatasetRequirements) -> DataPreparation:
        """Handle CIFAR-100 dataset"""

        dataset_name = "CIFAR100"
        cache_path = self.cache_dir / "cifar100"

        loader_code = f"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(
    root='{cache_path}',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR100(
    root='{cache_path}',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=True,
    num_workers={dataset_info.num_workers}
)

test_loader = DataLoader(
    test_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=False,
    num_workers={dataset_info.num_workers}
)

num_classes = 100
input_shape = (3, 32, 32)
"""

        return DataPreparation(
            dataset_name=dataset_name,
            cache_path=cache_path,
            loaders={},
            loader_code=loader_code,
            statistics={
                "num_classes": 100,
                "train_size": 50000,
                "test_size": 10000,
                "input_shape": (3, 32, 32),
            },
            samples=[],
        )

    async def _handle_mnist(self, dataset_info: DatasetRequirements) -> DataPreparation:
        """Handle MNIST dataset"""

        dataset_name = "MNIST"
        cache_path = self.cache_dir / "mnist"

        loader_code = f"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='{cache_path}',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='{cache_path}',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=True,
    num_workers={dataset_info.num_workers}
)

test_loader = DataLoader(
    test_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=False,
    num_workers={dataset_info.num_workers}
)

num_classes = 10
input_shape = (1, 28, 28)
"""

        return DataPreparation(
            dataset_name=dataset_name,
            cache_path=cache_path,
            loaders={},
            loader_code=loader_code,
            statistics={
                "num_classes": 10,
                "train_size": 60000,
                "test_size": 10000,
                "input_shape": (1, 28, 28),
            },
            samples=[],
        )

    async def _handle_imagenet(self, dataset_info: DatasetRequirements) -> DataPreparation:
        """Handle ImageNet dataset"""

        # ImageNet requires manual download
        self.logger.warning(
            "ImageNet requires manual download. Please download from official source."
        )

        dataset_name = "ImageNet"
        cache_path = self.cache_dir / "imagenet"

        loader_code = f"""
# ImageNet requires manual download
# Download from: https://image-net.org/

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assumes ImageNet is downloaded to {cache_path}
train_dataset = datasets.ImageNet(
    root='{cache_path}',
    split='train',
    transform=transform
)

val_dataset = datasets.ImageNet(
    root='{cache_path}',
    split='val',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=True,
    num_workers={dataset_info.num_workers}
)

val_loader = DataLoader(
    val_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=False,
    num_workers={dataset_info.num_workers}
)

num_classes = 1000
input_shape = (3, 224, 224)
"""

        return DataPreparation(
            dataset_name=dataset_name,
            cache_path=cache_path,
            loaders={},
            loader_code=loader_code,
            statistics={
                "num_classes": 1000,
                "train_size": 1281167,
                "val_size": 50000,
                "input_shape": (3, 224, 224),
            },
            samples=[],
        )

    async def _handle_coco(self, dataset_info: DatasetRequirements) -> DataPreparation:
        """Handle COCO dataset"""

        dataset_name = "COCO"
        cache_path = self.cache_dir / "coco"

        loader_code = f"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# COCO dataset
train_dataset = datasets.CocoDetection(
    root='{cache_path}/train2017',
    annFile='{cache_path}/annotations/instances_train2017.json',
    transform=transform
)

val_dataset = datasets.CocoDetection(
    root='{cache_path}/val2017',
    annFile='{cache_path}/annotations/instances_val2017.json',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=True,
    num_workers={dataset_info.num_workers},
    collate_fn=lambda x: x  # Custom collate for COCO
)

val_loader = DataLoader(
    val_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=False,
    num_workers={dataset_info.num_workers},
    collate_fn=lambda x: x
)

num_classes = 80  # COCO has 80 object categories
"""

        return DataPreparation(
            dataset_name=dataset_name,
            cache_path=cache_path,
            loaders={},
            loader_code=loader_code,
            statistics={
                "num_classes": 80,
                "train_size": 118287,
                "val_size": 5000,
            },
            samples=[],
        )

    async def _discover_and_prepare(
        self, dataset_info: DatasetRequirements
    ) -> DataPreparation:
        """
        Try to auto-discover dataset from common sources

        Args:
            dataset_info: Dataset requirements

        Returns:
            Data preparation result
        """

        dataset_name = dataset_info.name
        self.logger.warning(
            f"No built-in handler for {dataset_name}, generating generic loader"
        )

        cache_path = self.cache_dir / dataset_name.lower()

        # Generate generic loader code
        loader_code = f"""
# Generic data loader for {dataset_name}
# You may need to customize this for your specific dataset

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        # TODO: Implement data loading logic

    def __len__(self):
        # TODO: Return dataset size
        return 0

    def __getitem__(self, idx):
        # TODO: Implement item loading
        pass

# Create dataset instances
train_dataset = CustomDataset('{cache_path}/train')
test_dataset = CustomDataset('{cache_path}/test')

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=True,
    num_workers={dataset_info.num_workers}
)

test_loader = DataLoader(
    test_dataset,
    batch_size={dataset_info.batch_size},
    shuffle=False,
    num_workers={dataset_info.num_workers}
)
"""

        return DataPreparation(
            dataset_name=dataset_name,
            cache_path=cache_path,
            loaders={},
            loader_code=loader_code,
            statistics={"note": "Generic loader - customize as needed"},
            samples=[],
        )
