"""
HuggingFace datasets integration for SciAgent

Provides utilities for:
- Loading datasets from HuggingFace Hub
- Dataset preprocessing
- Dataset splitting
- Creating data loaders
- Dataset search and discovery
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a HuggingFace dataset"""

    dataset_id: str
    description: str
    downloads: int = 0
    likes: int = 0
    tasks: List[str] = None
    languages: List[str] = None
    size: Optional[str] = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.languages is None:
            self.languages = []


@dataclass
class DatasetLoadConfig:
    """Configuration for loading a HuggingFace dataset"""

    dataset_id: str
    split: Optional[str] = None  # "train", "test", "validation", or None for all
    subset: Optional[str] = None  # For datasets with multiple subsets
    cache_dir: Optional[Path] = None
    streaming: bool = False  # For large datasets
    num_samples: Optional[int] = None  # Limit number of samples


@dataclass
class DataLoaderConfig:
    """Configuration for creating PyTorch DataLoader"""

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False


class HuggingFaceDatasetManager:
    """
    Manager for HuggingFace datasets

    Provides high-level interface for:
    - Dataset discovery and search
    - Dataset loading and caching
    - Preprocessing and tokenization
    - Creating data loaders
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the dataset manager

        Args:
            cache_dir: Directory for caching datasets (default: ~/.cache/huggingface/datasets)
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_datasets: Dict[str, DatasetDict] = {}

        logger.info(f"Initialized HuggingFace dataset manager with cache: {self.cache_dir}")

    def search_datasets(
        self,
        query: str,
        task: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 10,
    ) -> List[DatasetInfo]:
        """
        Search for datasets on HuggingFace Hub

        Args:
            query: Search query
            task: Filter by task (e.g., "text-classification", "translation")
            language: Filter by language (e.g., "en", "multilingual")
            limit: Maximum number of results

        Returns:
            List of dataset information
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            datasets = api.list_datasets(
                search=query,
                task=task,
                limit=limit,
                sort="downloads",
                direction=-1,
            )

            results = []
            for dataset in datasets:
                dataset_info = DatasetInfo(
                    dataset_id=dataset.id,
                    description=getattr(dataset, 'description', ''),
                    downloads=getattr(dataset, 'downloads', 0),
                    likes=getattr(dataset, 'likes', 0),
                    tasks=list(getattr(dataset, 'tags', [])),
                )

                # Filter by language if specified
                if language is None or language in dataset_info.languages:
                    results.append(dataset_info)

            logger.info(f"Found {len(results)} datasets for query: {query}")
            return results[:limit]

        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            return self._get_fallback_datasets(query, task)

    def _get_fallback_datasets(self, query: str, task: Optional[str] = None) -> List[DatasetInfo]:
        """Get fallback datasets when API search fails"""

        common_datasets = {
            "text-classification": [
                ("imdb", "Movie review sentiment classification", 50000),
                ("ag_news", "News article classification", 127600),
                ("yelp_review_full", "Yelp review classification", 650000),
            ],
            "question-answering": [
                ("squad", "Stanford Question Answering Dataset", 100000),
                ("squad_v2", "SQuAD 2.0 with unanswerable questions", 150000),
            ],
            "translation": [
                ("wmt14", "Machine translation (WMT)", 4500000),
                ("opus100", "Parallel corpus for 100 languages", 55000000),
            ],
            "summarization": [
                ("cnn_dailymail", "CNN/DailyMail summarization", 300000),
                ("xsum", "Extreme summarization", 227000),
            ],
        }

        if task and task in common_datasets:
            return [
                DatasetInfo(dataset_id=did, description=desc, downloads=downloads, tasks=[task])
                for did, desc, downloads in common_datasets[task]
            ]

        # Return general purpose datasets
        return [
            DatasetInfo(
                dataset_id="glue",
                description="General Language Understanding Evaluation benchmark",
                downloads=100000,
                tasks=["text-classification"],
            ),
            DatasetInfo(
                dataset_id="imdb",
                description="Movie review sentiment",
                downloads=50000,
                tasks=["text-classification"],
            ),
        ]

    def load_dataset(
        self,
        config: DatasetLoadConfig,
    ) -> Union[Dataset, DatasetDict]:
        """
        Load a HuggingFace dataset

        Args:
            config: Dataset loading configuration

        Returns:
            Dataset or DatasetDict
        """
        cache_key = f"{config.dataset_id}:{config.subset}:{config.split}"

        # Return cached dataset if available
        if cache_key in self.loaded_datasets:
            logger.info(f"Using cached dataset: {config.dataset_id}")
            dataset = self.loaded_datasets[cache_key]
        else:
            logger.info(f"Loading dataset: {config.dataset_id}")

            try:
                dataset = load_dataset(
                    config.dataset_id,
                    name=config.subset,
                    split=config.split,
                    cache_dir=config.cache_dir or self.cache_dir,
                    streaming=config.streaming,
                )

                # Cache the dataset
                if not config.streaming:
                    self.loaded_datasets[cache_key] = dataset

            except Exception as e:
                logger.error(f"Error loading dataset {config.dataset_id}: {e}")
                raise

        # Limit number of samples if specified
        if config.num_samples and not config.streaming:
            if isinstance(dataset, DatasetDict):
                dataset = DatasetDict({
                    split: ds.select(range(min(config.num_samples, len(ds))))
                    for split, ds in dataset.items()
                })
            else:
                dataset = dataset.select(range(min(config.num_samples, len(dataset))))

        logger.info(f"Successfully loaded dataset: {config.dataset_id}")
        return dataset

    def preprocess_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        preprocessing_fn: Callable,
        batched: bool = True,
        remove_columns: Optional[List[str]] = None,
        num_proc: Optional[int] = None,
    ) -> Union[Dataset, DatasetDict]:
        """
        Preprocess a dataset with a custom function

        Args:
            dataset: Dataset to preprocess
            preprocessing_fn: Function to apply to each example
            batched: Whether to process in batches
            remove_columns: Columns to remove after preprocessing
            num_proc: Number of processes for parallel processing

        Returns:
            Preprocessed dataset
        """
        logger.info("Preprocessing dataset...")

        try:
            processed_dataset = dataset.map(
                preprocessing_fn,
                batched=batched,
                remove_columns=remove_columns,
                num_proc=num_proc,
            )

            logger.info("Dataset preprocessing completed")
            return processed_dataset

        except Exception as e:
            logger.error(f"Error preprocessing dataset: {e}")
            raise

    def tokenize_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        tokenizer: Any,
        text_column: str = "text",
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> Union[Dataset, DatasetDict]:
        """
        Tokenize a text dataset

        Args:
            dataset: Dataset to tokenize
            tokenizer: HuggingFace tokenizer
            text_column: Name of the text column
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            padding: Padding strategy

        Returns:
            Tokenized dataset
        """
        logger.info(f"Tokenizing dataset (max_length={max_length})...")

        def tokenize_fn(examples):
            return tokenizer(
                examples[text_column],
                truncation=truncation,
                padding=padding,
                max_length=max_length,
            )

        return self.preprocess_dataset(
            dataset,
            tokenize_fn,
            batched=True,
        )

    def create_dataloader(
        self,
        dataset: Dataset,
        config: DataLoaderConfig,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from a dataset

        Args:
            dataset: Dataset to create loader from
            config: DataLoader configuration
            collate_fn: Custom collate function (optional)

        Returns:
            PyTorch DataLoader
        """
        logger.info(f"Creating DataLoader (batch_size={config.batch_size})...")

        # Set format to PyTorch tensors
        dataset.set_format(type="torch")

        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            collate_fn=collate_fn,
        )

        logger.info(f"DataLoader created with {len(dataloader)} batches")
        return dataloader

    def split_dataset(
        self,
        dataset: Dataset,
        train_size: float = 0.8,
        test_size: Optional[float] = None,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Split a dataset into train/test or train/val/test

        Args:
            dataset: Dataset to split
            train_size: Proportion for training (0-1)
            test_size: Proportion for test (0-1), if None uses remaining after train
            seed: Random seed

        Returns:
            DatasetDict with splits
        """
        logger.info(f"Splitting dataset (train={train_size}, test={test_size})...")

        if test_size is None:
            # Binary split
            split_dataset = dataset.train_test_split(
                train_size=train_size,
                seed=seed,
            )
            return split_dataset
        else:
            # Three-way split
            val_size = 1.0 - train_size - test_size
            assert val_size > 0, "train_size + test_size must be < 1.0"

            # First split: train vs (val + test)
            train_val_test = dataset.train_test_split(
                train_size=train_size,
                seed=seed,
            )

            # Second split: val vs test
            val_test_split = train_val_test["test"].train_test_split(
                train_size=val_size / (val_size + test_size),
                seed=seed,
            )

            return DatasetDict({
                "train": train_val_test["train"],
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            })

    def get_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """
        Get information about a loaded dataset

        Args:
            dataset: Dataset to get info for

        Returns:
            Dictionary with dataset information
        """
        if isinstance(dataset, DatasetDict):
            info = {
                "type": "DatasetDict",
                "splits": {
                    split: {
                        "num_rows": len(ds),
                        "num_columns": len(ds.column_names),
                        "columns": ds.column_names,
                        "features": str(ds.features),
                    }
                    for split, ds in dataset.items()
                },
            }
        else:
            info = {
                "type": "Dataset",
                "num_rows": len(dataset),
                "num_columns": len(dataset.column_names),
                "columns": dataset.column_names,
                "features": str(dataset.features),
            }

        return info

    def get_dataset_code_template(
        self,
        dataset_id: str,
        split: str = "train",
        subset: Optional[str] = None,
        include_preprocessing: bool = True,
    ) -> str:
        """
        Generate code template for using a HuggingFace dataset

        Args:
            dataset_id: Dataset identifier
            split: Dataset split
            subset: Dataset subset (if applicable)
            include_preprocessing: Whether to include preprocessing example

        Returns:
            Python code template as string
        """
        subset_arg = f', name="{subset}"' if subset else ''

        template = f'''
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("{dataset_id}"{subset_arg}, split="{split}")

print(f"Dataset size: {{len(dataset)}}")
print(f"Features: {{dataset.features}}")
print(f"Example: {{dataset[0]}}")
'''

        if include_preprocessing:
            template += '''
# Preprocess dataset
def preprocess_function(examples):
    # Add your preprocessing logic here
    # For example, tokenization:
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # return tokenizer(examples["text"], truncation=True, padding="max_length")
    return examples

processed_dataset = dataset.map(preprocess_function, batched=True)
'''

        template += '''
# Create DataLoader
from torch.utils.data import DataLoader

processed_dataset.set_format(type="torch")
dataloader = DataLoader(processed_dataset, batch_size=32, shuffle=True)

# Train your model
for batch in dataloader:
    # Your training logic here
    pass
'''

        return template

    def clear_cache(self) -> None:
        """Clear all loaded datasets from memory"""
        self.loaded_datasets.clear()
        logger.info("Cleared dataset cache")

    def get_popular_datasets(self) -> List[DatasetInfo]:
        """Get a list of popular datasets for common tasks"""

        return [
            DatasetInfo(
                dataset_id="imdb",
                description="Movie review sentiment classification (binary)",
                downloads=50000,
                tasks=["text-classification"],
                languages=["en"],
            ),
            DatasetInfo(
                dataset_id="squad",
                description="Stanford Question Answering Dataset",
                downloads=100000,
                tasks=["question-answering"],
                languages=["en"],
            ),
            DatasetInfo(
                dataset_id="cnn_dailymail",
                description="News article summarization",
                downloads=30000,
                tasks=["summarization"],
                languages=["en"],
            ),
            DatasetInfo(
                dataset_id="wmt14",
                description="Machine translation (multiple language pairs)",
                downloads=40000,
                tasks=["translation"],
                languages=["multilingual"],
            ),
            DatasetInfo(
                dataset_id="glue",
                description="General Language Understanding Evaluation",
                downloads=80000,
                tasks=["text-classification", "natural-language-inference"],
                languages=["en"],
            ),
            DatasetInfo(
                dataset_id="super_glue",
                description="Advanced language understanding tasks",
                downloads=50000,
                tasks=["text-classification", "question-answering"],
                languages=["en"],
            ),
            DatasetInfo(
                dataset_id="conll2003",
                description="Named Entity Recognition",
                downloads=30000,
                tasks=["token-classification"],
                languages=["en"],
            ),
            DatasetInfo(
                dataset_id="multi_news",
                description="Multi-document summarization",
                downloads=10000,
                tasks=["summarization"],
                languages=["en"],
            ),
        ]
