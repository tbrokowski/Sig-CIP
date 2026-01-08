"""
HuggingFace Integration Examples for SciAgent

Demonstrates:
- Loading and using HuggingFace models
- Loading and using HuggingFace datasets
- Fine-tuning pre-trained models
- Running experiments with HuggingFace components
"""

import asyncio
from pathlib import Path

from sciagent import SciAgent
from sciagent.integrations.huggingface import HuggingFaceModelManager, HuggingFaceDatasetManager
from sciagent.integrations.huggingface.models import ModelLoadConfig, FineTuneConfig
from sciagent.integrations.huggingface.datasets import DatasetLoadConfig, DataLoaderConfig


async def example_1_search_models():
    """
    Example 1: Search for HuggingFace models
    """
    print("=" * 70)
    print("Example 1: Searching HuggingFace Models")
    print("=" * 70)

    manager = HuggingFaceModelManager()

    # Search for sentiment analysis models
    print("\n1. Searching for sentiment analysis models...")
    models = manager.search_models(
        query="sentiment",
        task="text-classification",
        limit=5
    )

    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model.model_id}")
        print(f"   Task: {model.task}")
        print(f"   Description: {model.description[:80]}...")
        print(f"   Downloads: {model.downloads:,}")

    # Search for question answering models
    print("\n2. Searching for question answering models...")
    models = manager.search_models(
        query="question answering",
        task="question-answering",
        limit=3
    )

    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model.model_id}")
        print(f"   Task: {model.task}")


async def example_2_load_and_use_model():
    """
    Example 2: Load and use a HuggingFace model
    """
    print("\n" + "=" * 70)
    print("Example 2: Loading and Using a Model")
    print("=" * 70)

    manager = HuggingFaceModelManager()

    # Create a text classification pipeline (simple approach)
    print("\n1. Using pipeline API (recommended for inference)...")
    pipe = manager.create_pipeline(
        task="text-classification",
        model_id="distilbert-base-uncased-finetuned-sst-2-english",
        device=None,  # CPU
    )

    # Test the pipeline
    texts = [
        "I love this product! It's amazing!",
        "This is terrible. Very disappointed.",
        "It's okay, nothing special."
    ]

    print("\nSentiment analysis results:")
    for text in texts:
        result = pipe(text)[0]
        print(f"\nText: {text}")
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")

    # Load model manually (for fine-tuning or custom usage)
    print("\n\n2. Loading model manually...")
    config = ModelLoadConfig(
        model_id="bert-base-uncased",
        task="fill-mask",
        device="cpu",
    )

    model_dict = manager.load_model(config)
    print(f"Loaded model: {model_dict['model_id']}")
    print(f"Device: {model_dict['device']}")
    print(f"Model type: {type(model_dict['model']).__name__}")


async def example_3_search_datasets():
    """
    Example 3: Search for HuggingFace datasets
    """
    print("\n" + "=" * 70)
    print("Example 3: Searching HuggingFace Datasets")
    print("=" * 70)

    manager = HuggingFaceDatasetManager()

    # Search for sentiment datasets
    print("\n1. Searching for sentiment datasets...")
    datasets = manager.search_datasets(
        query="sentiment",
        task="text-classification",
        limit=5
    )

    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset.dataset_id}")
        print(f"   Description: {dataset.description[:80]}...")
        print(f"   Downloads: {dataset.downloads:,}")
        print(f"   Tasks: {', '.join(dataset.tasks)}")

    # Get popular datasets
    print("\n\n2. Popular datasets:")
    popular = manager.get_popular_datasets()

    for i, dataset in enumerate(popular[:5], 1):
        print(f"\n{i}. {dataset.dataset_id}")
        print(f"   Tasks: {', '.join(dataset.tasks)}")
        print(f"   Languages: {', '.join(dataset.languages)}")


async def example_4_load_and_use_dataset():
    """
    Example 4: Load and use a HuggingFace dataset
    """
    print("\n" + "=" * 70)
    print("Example 4: Loading and Using a Dataset")
    print("=" * 70)

    manager = HuggingFaceDatasetManager()

    # Load IMDB dataset
    print("\n1. Loading IMDB dataset...")
    config = DatasetLoadConfig(
        dataset_id="imdb",
        split="train",
        num_samples=100,  # Limit to 100 samples for demo
    )

    dataset = manager.load_dataset(config)

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    print(f"\nFirst example:")
    print(f"Text: {dataset[0]['text'][:100]}...")
    print(f"Label: {dataset[0]['label']}")

    # Get dataset info
    print("\n2. Dataset information:")
    info = manager.get_dataset_info(dataset)
    for key, value in info.items():
        print(f"{key}: {value}")


async def example_5_preprocess_dataset():
    """
    Example 5: Preprocess and tokenize a dataset
    """
    print("\n" + "=" * 70)
    print("Example 5: Preprocessing and Tokenizing Dataset")
    print("=" * 70)

    from transformers import AutoTokenizer

    manager = HuggingFaceDatasetManager()

    # Load dataset
    print("\n1. Loading dataset...")
    config = DatasetLoadConfig(
        dataset_id="imdb",
        split="train",
        num_samples=50,
    )

    dataset = manager.load_dataset(config)

    # Tokenize dataset
    print("\n2. Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenized_dataset = manager.tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        text_column="text",
        max_length=128,
    )

    print(f"Tokenized dataset: {len(tokenized_dataset)} samples")
    print(f"Columns: {tokenized_dataset.column_names}")
    print(f"\nFirst tokenized example (input_ids): {tokenized_dataset[0]['input_ids'][:10]}...")

    # Create DataLoader
    print("\n3. Creating PyTorch DataLoader...")
    dataloader_config = DataLoaderConfig(
        batch_size=8,
        shuffle=True,
    )

    dataloader = manager.create_dataloader(
        dataset=tokenized_dataset,
        config=dataloader_config,
    )

    print(f"DataLoader created with {len(dataloader)} batches")

    # Show a batch
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")


async def example_6_sciagent_with_huggingface():
    """
    Example 6: Run SciAgent experiment with HuggingFace dataset
    """
    print("\n" + "=" * 70)
    print("Example 6: SciAgent with HuggingFace Dataset")
    print("=" * 70)

    agent = SciAgent()

    # Run experiment using HuggingFace dataset
    # Use "hf:dataset_id" format to specify HuggingFace dataset
    query = """
Test if fine-tuning BERT on IMDB improves sentiment classification:
- Use HuggingFace IMDB dataset (hf:imdb)
- Fine-tune distilbert-base-uncased
- Compare with baseline
- Measure accuracy and F1 score
- Use train/test split
"""

    print("\nRunning experiment with HuggingFace dataset...")
    print("Query:", query.strip())

    # Note: This would run a full experiment
    # For demo, we'll just show the concept
    print("\nTo run this experiment, use:")
    print("  result = await agent.run(query, interactive=False)")
    print("\nThe Data Agent will:")
    print("  1. Detect 'hf:imdb' dataset reference")
    print("  2. Load IMDB from HuggingFace")
    print("  3. Generate appropriate data loaders")
    print("\nThe Coding Agent will:")
    print("  1. Detect BERT/fine-tuning task")
    print("  2. Suggest HuggingFace model templates")
    print("  3. Generate code using transformers library")


async def example_7_model_code_templates():
    """
    Example 7: Generate code templates for HuggingFace models
    """
    print("\n" + "=" * 70)
    print("Example 7: Model Code Templates")
    print("=" * 70)

    manager = HuggingFaceModelManager()

    # Get template for text classification
    print("\n1. Text classification pipeline template:")
    template = manager.get_model_code_template(
        model_id="distilbert-base-uncased-finetuned-sst-2-english",
        task="text-classification",
        use_pipeline=True
    )
    print(template)

    # Get template for manual model loading
    print("\n2. Manual model loading template:")
    template = manager.get_model_code_template(
        model_id="bert-base-uncased",
        task="text-classification",
        use_pipeline=False
    )
    print(template)


async def example_8_dataset_code_templates():
    """
    Example 8: Generate code templates for HuggingFace datasets
    """
    print("\n" + "=" * 70)
    print("Example 8: Dataset Code Templates")
    print("=" * 70)

    manager = HuggingFaceDatasetManager()

    # Get template for dataset loading and preprocessing
    template = manager.get_dataset_code_template(
        dataset_id="imdb",
        split="train",
        include_preprocessing=True
    )

    print("\nDataset loading and preprocessing template:")
    print(template)


async def example_9_complete_workflow():
    """
    Example 9: Complete workflow with HuggingFace components
    """
    print("\n" + "=" * 70)
    print("Example 9: Complete Workflow")
    print("=" * 70)

    print("\nThis example demonstrates a complete ML workflow:")
    print("1. Search for appropriate model and dataset")
    print("2. Load and preprocess data")
    print("3. Load pre-trained model")
    print("4. Create training pipeline")
    print("5. Evaluate model")

    # Initialize managers
    model_manager = HuggingFaceModelManager()
    dataset_manager = HuggingFaceDatasetManager()

    # Step 1: Search
    print("\n1. Searching for model and dataset...")
    models = model_manager.search_models("sentiment", task="text-classification", limit=1)
    datasets = dataset_manager.search_datasets("imdb", limit=1)

    print(f"   Selected model: {models[0].model_id}")
    print(f"   Selected dataset: {datasets[0].dataset_id}")

    # Step 2: Load dataset
    print("\n2. Loading dataset...")
    dataset_config = DatasetLoadConfig(
        dataset_id="imdb",
        split="train",
        num_samples=100,  # Small sample for demo
    )
    dataset = dataset_manager.load_dataset(dataset_config)
    print(f"   Loaded {len(dataset)} samples")

    # Step 3: Load model
    print("\n3. Loading model...")
    pipe = model_manager.create_pipeline(
        task="text-classification",
        model_id=models[0].model_id,
    )
    print(f"   Model loaded: {models[0].model_id}")

    # Step 4: Inference
    print("\n4. Running inference on sample...")
    sample_text = dataset[0]['text'][:100]
    result = pipe(sample_text)[0]
    print(f"   Text: {sample_text}...")
    print(f"   Prediction: {result['label']} (confidence: {result['score']:.4f})")

    # Step 5: Statistics
    print("\n5. Dataset statistics:")
    info = dataset_manager.get_dataset_info(dataset)
    print(f"   Type: {info['type']}")
    print(f"   Samples: {info['num_rows']}")
    print(f"   Features: {', '.join(info['columns'])}")


async def main():
    """Run all examples"""

    print("\n")
    print("*" * 70)
    print("HuggingFace Integration Examples for SciAgent")
    print("*" * 70)

    # Run each example
    await example_1_search_models()
    await example_2_load_and_use_model()
    await example_3_search_datasets()
    await example_4_load_and_use_dataset()
    await example_5_preprocess_dataset()
    await example_6_sciagent_with_huggingface()
    await example_7_model_code_templates()
    await example_8_dataset_code_templates()
    await example_9_complete_workflow()

    print("\n")
    print("*" * 70)
    print("All examples completed!")
    print("*" * 70)
    print("\nKey Takeaways:")
    print("- HuggingFace models and datasets are fully integrated with SciAgent")
    print("- Use 'hf:dataset_id' format in experiments to use HuggingFace datasets")
    print("- SciAgent automatically suggests appropriate models for tasks")
    print("- Both pipeline API and manual loading are supported")
    print("- Preprocessing and tokenization utilities are available")
    print("\nFor more information:")
    print("- HuggingFace Models: https://huggingface.co/models")
    print("- HuggingFace Datasets: https://huggingface.co/datasets")
    print("- Transformers docs: https://huggingface.co/docs/transformers")
    print()


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())

    # Or run individual examples:
    # asyncio.run(example_1_search_models())
    # asyncio.run(example_2_load_and_use_model())
    # etc.
