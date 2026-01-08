# HuggingFace Integration Guide

SciAgent provides seamless integration with HuggingFace's ecosystem of pre-trained models and datasets, enabling you to leverage state-of-the-art machine learning models and access thousands of datasets for your experiments.

## Table of Contents

1. [Overview](#overview)
2. [HuggingFace Models](#huggingface-models)
3. [HuggingFace Datasets](#huggingface-datasets)
4. [Integration with SciAgent](#integration-with-sciagent)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

### What's Included

- **Model Manager**: Search, load, and use pre-trained models from HuggingFace Hub
- **Dataset Manager**: Access and preprocess datasets from HuggingFace Datasets
- **Automatic Integration**: SciAgent automatically suggests appropriate models for tasks
- **Code Generation**: Generate ready-to-use code templates
- **Fine-tuning Support**: Built-in support for model fine-tuning

### Prerequisites

```bash
# All required packages are included in SciAgent
pip install sciagent

# Or install individually
pip install transformers datasets torch
```

## HuggingFace Models

### Model Manager

The `HuggingFaceModelManager` provides a high-level interface for working with models:

```python
from sciagent.integrations.huggingface import HuggingFaceModelManager

manager = HuggingFaceModelManager()
```

### Searching for Models

Search for models by keyword and task:

```python
# Search for sentiment analysis models
models = manager.search_models(
    query="sentiment",
    task="text-classification",
    limit=10
)

for model in models:
    print(f"{model.model_id} - {model.description}")
    print(f"  Downloads: {model.downloads:,}")
    print(f"  Task: {model.task}")
```

**Supported Tasks:**
- `text-classification` - Sentiment analysis, topic classification
- `text-generation` - Text completion, story generation
- `question-answering` - Answer questions from context
- `summarization` - Text summarization
- `translation` - Machine translation
- `token-classification` - NER, POS tagging
- `image-classification` - Image classification
- `object-detection` - Object detection in images
- And many more...

### Loading Models

#### Pipeline API (Recommended for Inference)

The simplest way to use models for inference:

```python
# Create a pipeline
pipe = manager.create_pipeline(
    task="text-classification",
    model_id="distilbert-base-uncased-finetuned-sst-2-english",
    device=0,  # Use GPU 0, or None for CPU
)

# Use the pipeline
result = pipe("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Benefits:**
- Simple, one-line usage
- Automatic preprocessing
- Optimized for inference
- No need to handle tokenization manually

#### Manual Model Loading (For Fine-tuning or Custom Usage)

For more control:

```python
from sciagent.integrations.huggingface.models import ModelLoadConfig

config = ModelLoadConfig(
    model_id="bert-base-uncased",
    task="text-classification",
    device="cuda",  # or "cpu", "mps", None for auto
    quantization="8bit",  # Optional: "8bit", "4bit", None
)

model_dict = manager.load_model(config)

model = model_dict["model"]
tokenizer = model_dict["tokenizer"]
device = model_dict["device"]

# Now you can use model and tokenizer directly
inputs = tokenizer("Hello world", return_tensors="pt").to(device)
outputs = model(**inputs)
```

**Model Loading Options:**
- `device`: Auto-selects CUDA > MPS > CPU if None
- `quantization`: Reduce model size with 8-bit or 4-bit quantization
- `use_fast_tokenizer`: Use fast Rust-based tokenizers (default: True)
- `trust_remote_code`: Allow loading models with custom code
- `cache_dir`: Custom cache directory

### Fine-tuning Models

SciAgent provides built-in support for fine-tuning:

```python
from sciagent.integrations.huggingface.models import FineTuneConfig

# Prepare your datasets (HuggingFace datasets format)
# train_dataset = ...
# eval_dataset = ...

config = FineTuneConfig(
    output_dir=Path("./my_model"),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    logging_steps=100,
    eval_steps=500,
)

result = await manager.fine_tune(
    model_id="bert-base-uncased",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config=config,
)

print(f"Training completed! Model saved to: {result['model_path']}")
```

### Code Templates

Generate ready-to-use code:

```python
# Generate pipeline code
code = manager.get_model_code_template(
    model_id="distilbert-base-uncased-finetuned-sst-2-english",
    task="text-classification",
    use_pipeline=True
)

print(code)
```

Output:
```python
from transformers import pipeline

# Create pipeline for text-classification
pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Example usage
result = pipe("Your input text here")
print(result)
```

## HuggingFace Datasets

### Dataset Manager

The `HuggingFaceDatasetManager` provides tools for working with datasets:

```python
from sciagent.integrations.huggingface import HuggingFaceDatasetManager

manager = HuggingFaceDatasetManager()
```

### Searching for Datasets

```python
# Search for datasets
datasets = manager.search_datasets(
    query="sentiment",
    task="text-classification",
    language="en",
    limit=10
)

for dataset in datasets:
    print(f"{dataset.dataset_id} - {dataset.description}")
    print(f"  Downloads: {dataset.downloads:,}")
    print(f"  Tasks: {', '.join(dataset.tasks)}")
```

### Popular Datasets

Get a curated list of popular datasets:

```python
popular = manager.get_popular_datasets()

for dataset in popular:
    print(f"{dataset.dataset_id}")
    print(f"  Tasks: {', '.join(dataset.tasks)}")
    print(f"  Languages: {', '.join(dataset.languages)}")
```

**Popular datasets include:**
- `imdb` - Movie review sentiment
- `squad` - Question answering
- `cnn_dailymail` - Summarization
- `wmt14` - Translation
- `glue` - Language understanding benchmark
- `conll2003` - Named entity recognition

### Loading Datasets

```python
from sciagent.integrations.huggingface.datasets import DatasetLoadConfig

config = DatasetLoadConfig(
    dataset_id="imdb",
    split="train",  # "train", "test", "validation", or None for all
    subset=None,  # For datasets with multiple configurations
    num_samples=1000,  # Optional: limit number of samples
    streaming=False,  # True for large datasets
)

dataset = manager.load_dataset(config)

print(f"Loaded {len(dataset)} samples")
print(f"Columns: {dataset.column_names}")
print(f"First example: {dataset[0]}")
```

### Preprocessing Datasets

#### Custom Preprocessing

```python
def preprocess_function(examples):
    # Your custom preprocessing
    examples["text"] = [text.lower() for text in examples["text"]]
    return examples

processed_dataset = manager.preprocess_dataset(
    dataset=dataset,
    preprocessing_fn=preprocess_function,
    batched=True,
    num_proc=4,  # Parallel processing
)
```

#### Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokenized_dataset = manager.tokenize_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    text_column="text",
    max_length=512,
    truncation=True,
    padding="max_length",
)

print(f"Tokenized {len(tokenized_dataset)} samples")
```

### Creating DataLoaders

Convert datasets to PyTorch DataLoaders:

```python
from sciagent.integrations.huggingface.datasets import DataLoaderConfig

config = DataLoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # Faster data transfer to GPU
)

dataloader = manager.create_dataloader(
    dataset=tokenized_dataset,
    config=config,
)

# Use in training loop
for batch in dataloader:
    # batch contains PyTorch tensors
    input_ids = batch["input_ids"]
    labels = batch["label"]
    # ... your training code
```

### Splitting Datasets

Split datasets into train/val/test:

```python
# Binary split (train/test)
split_dataset = manager.split_dataset(
    dataset=dataset,
    train_size=0.8,
    seed=42,
)

print(f"Train: {len(split_dataset['train'])} samples")
print(f"Test: {len(split_dataset['test'])} samples")

# Three-way split (train/val/test)
split_dataset = manager.split_dataset(
    dataset=dataset,
    train_size=0.7,
    test_size=0.15,  # Validation will be 0.15
    seed=42,
)

print(f"Train: {len(split_dataset['train'])} samples")
print(f"Validation: {len(split_dataset['validation'])} samples")
print(f"Test: {len(split_dataset['test'])} samples")
```

## Integration with SciAgent

### Using HuggingFace Datasets in Experiments

Use the `hf:` prefix to specify HuggingFace datasets:

```python
from sciagent import SciAgent

agent = SciAgent()

result = await agent.run("""
Test if fine-tuning BERT improves sentiment classification:
- Use HuggingFace IMDB dataset (hf:imdb)
- Fine-tune bert-base-uncased
- Compare with distilbert baseline
- Measure accuracy and F1 score
""", interactive=False)
```

**The Data Agent will automatically:**
1. Detect the `hf:imdb` reference
2. Load the IMDB dataset from HuggingFace
3. Generate appropriate data loaders
4. Provide dataset statistics

### Automatic Model Suggestions

The Coding Agent automatically suggests HuggingFace models based on your task:

```python
result = await agent.run("""
Build a sentiment classifier for movie reviews
""", interactive=False)
```

**The Coding Agent will:**
1. Detect it's a sentiment classification task
2. Suggest appropriate models like `distilbert-base-uncased-finetuned-sst-2-english`
3. Generate code using HuggingFace transformers
4. Include both pipeline and fine-tuning examples

### Task-Specific Suggestions

SciAgent provides intelligent model suggestions for various tasks:

#### NLP Tasks

**Text Classification / Sentiment Analysis:**
```python
# Suggested models:
# - distilbert-base-uncased-finetuned-sst-2-english
# - bert-base-uncased
# - roberta-base
```

**Question Answering:**
```python
# Suggested models:
# - distilbert-base-cased-distilled-squad
# - bert-large-uncased-whole-word-masking-finetuned-squad
```

**Summarization:**
```python
# Suggested models:
# - facebook/bart-large-cnn
# - t5-base
# - google/pegasus-xsum
```

**Translation:**
```python
# Suggested models:
# - t5-base
# - Helsinki-NLP/opus-mt-en-de
# - facebook/mbart-large-50-many-to-many-mmt
```

#### Computer Vision Tasks

**Image Classification:**
```python
# Suggested models:
# - google/vit-base-patch16-224
# - microsoft/resnet-50
# - facebook/deit-base-distilled-patch16-224
```

#### Multimodal Tasks

**Vision-Language:**
```python
# Suggested models:
# - openai/clip-vit-base-patch32
# - laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

## Advanced Usage

### Quantization for Large Models

Reduce memory usage with quantization:

```python
from sciagent.integrations.huggingface.models import ModelLoadConfig

# 8-bit quantization (reduce memory by ~4x)
config = ModelLoadConfig(
    model_id="facebook/opt-6.7b",
    quantization="8bit",
)

model_dict = manager.load_model(config)

# 4-bit quantization (reduce memory by ~8x)
config = ModelLoadConfig(
    model_id="facebook/opt-6.7b",
    quantization="4bit",
)

model_dict = manager.load_model(config)
```

### Streaming Large Datasets

For datasets too large to fit in memory:

```python
from sciagent.integrations.huggingface.datasets import DatasetLoadConfig

config = DatasetLoadConfig(
    dataset_id="c4",  # Large dataset
    split="train",
    streaming=True,  # Enable streaming
)

dataset = manager.load_dataset(config)

# Iterate through dataset without loading everything
for example in dataset.take(100):
    print(example)
```

### Custom Collate Functions

For complex batching:

```python
def custom_collate_fn(batch):
    # Custom batching logic
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    # Process as needed
    return {"texts": texts, "labels": labels}

dataloader = manager.create_dataloader(
    dataset=dataset,
    config=config,
    collate_fn=custom_collate_fn,
)
```

### Multi-GPU Training

```python
# Load model on specific GPU
config = ModelLoadConfig(
    model_id="bert-base-uncased",
    device="cuda:0",  # GPU 0
)

model_dict = manager.load_model(config)

# For multi-GPU, use PyTorch DataParallel or DistributedDataParallel
import torch.nn as nn

model = nn.DataParallel(model_dict["model"])
```

## Best Practices

### 1. Cache Management

Models and datasets are cached by default:

```python
# Default cache locations:
# - Models: ~/.cache/huggingface/hub
# - Datasets: ~/.cache/huggingface/datasets

# Custom cache directory:
model_manager = HuggingFaceModelManager(cache_dir=Path("/custom/cache"))
dataset_manager = HuggingFaceDatasetManager(cache_dir=Path("/custom/cache"))
```

### 2. Memory Management

```python
# Unload models when done
manager.unload_model("bert-base-uncased")

# Clear all cached models
manager.clear_cache()

# Use quantization for large models
config = ModelLoadConfig(
    model_id="facebook/opt-6.7b",
    quantization="8bit",
)
```

### 3. Reproducibility

```python
# Set random seeds
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Use fixed seed in dataset splitting
split_dataset = manager.split_dataset(dataset, train_size=0.8, seed=42)
```

### 4. Efficient Data Loading

```python
# Use multiple workers
config = DataLoaderConfig(
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
)

# Use batched preprocessing
processed = manager.preprocess_dataset(
    dataset=dataset,
    preprocessing_fn=preprocess_fn,
    batched=True,  # Process multiple examples at once
    num_proc=4,  # Parallel processing
)
```

### 5. Error Handling

```python
try:
    model_dict = manager.load_model(config)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Fallback to a different model
    config.model_id = "distilbert-base-uncased"
    model_dict = manager.load_model(config)
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Problem:** GPU out of memory when loading large models

**Solutions:**
```python
# Use quantization
config = ModelLoadConfig(model_id="large-model", quantization="8bit")

# Use CPU
config = ModelLoadConfig(model_id="large-model", device="cpu")

# Reduce batch size
dataloader_config = DataLoaderConfig(batch_size=8)  # Instead of 32
```

#### 2. Slow Dataset Loading

**Problem:** Dataset loading is very slow

**Solutions:**
```python
# Use streaming for large datasets
config = DatasetLoadConfig(dataset_id="c4", streaming=True)

# Increase num_workers
dataloader_config = DataLoaderConfig(batch_size=32, num_workers=8)

# Use smaller subset for testing
config = DatasetLoadConfig(dataset_id="imdb", num_samples=1000)
```

#### 3. Tokenization Issues

**Problem:** Tokenizer doesn't match model

**Solution:**
```python
# Always use matching tokenizer
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

#### 4. Dataset Not Found

**Problem:** Dataset ID not found on HuggingFace

**Solutions:**
```python
# Search for dataset first
datasets = manager.search_datasets("dataset_name", limit=5)
for ds in datasets:
    print(ds.dataset_id)

# Use correct dataset ID
config = DatasetLoadConfig(dataset_id="correct-dataset-id")
```

#### 5. Model Download Fails

**Problem:** Cannot download model from HuggingFace Hub

**Solutions:**
```bash
# Check internet connection
# Check HuggingFace Hub status: https://status.huggingface.co

# Use mirror (if in restricted region)
export HF_ENDPOINT=https://hf-mirror.com

# Manually download and specify local path
from transformers import AutoModel
model = AutoModel.from_pretrained("/path/to/local/model")
```

### Performance Tips

1. **Use Fast Tokenizers:** Set `use_fast_tokenizer=True` (default)
2. **Enable Pin Memory:** Use `pin_memory=True` in DataLoader for GPU training
3. **Batch Processing:** Use `batched=True` in dataset preprocessing
4. **Cache Datasets:** Datasets are cached automatically, reuse them
5. **Profile Code:** Use `torch.profiler` to identify bottlenecks

### Getting Help

1. **HuggingFace Documentation:**
   - Models: https://huggingface.co/docs/transformers
   - Datasets: https://huggingface.co/docs/datasets

2. **SciAgent Issues:**
   - GitHub: https://github.com/yourusername/sciagent/issues

3. **HuggingFace Forums:**
   - https://discuss.huggingface.co

4. **Examples:**
   - See `examples/huggingface_integration.py` for complete examples

## Summary

SciAgent's HuggingFace integration provides:

✅ **Easy Model Access:** Search, load, and use 100,000+ models
✅ **Dataset Integration:** Access 50,000+ datasets
✅ **Automatic Suggestions:** Intelligent model recommendations
✅ **Code Generation:** Ready-to-use templates
✅ **Fine-tuning Support:** Built-in training utilities
✅ **Production Ready:** Memory management, quantization, streaming
✅ **Seamless Integration:** Works naturally with SciAgent workflows

Start using HuggingFace with SciAgent today to accelerate your research and development!
