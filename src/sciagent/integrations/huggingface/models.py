"""
HuggingFace model integration for SciAgent

Provides utilities for:
- Loading pre-trained models from HuggingFace Hub
- Fine-tuning models
- Model evaluation
- Model deployment
- Model search and discovery
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a HuggingFace model"""

    model_id: str
    task: str
    description: str
    downloads: int = 0
    likes: int = 0
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ModelLoadConfig:
    """Configuration for loading a HuggingFace model"""

    model_id: str
    task: Optional[str] = None
    device: Optional[str] = None  # "cuda", "cpu", "mps", or None for auto
    quantization: Optional[str] = None  # "8bit", "4bit", None
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: Optional[Path] = None


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning a model"""

    output_dir: Path
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False
    gradient_accumulation_steps: int = 1


class HuggingFaceModelManager:
    """
    Manager for HuggingFace models

    Provides high-level interface for:
    - Model discovery and search
    - Model loading and caching
    - Fine-tuning
    - Inference
    - Model evaluation
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the model manager

        Args:
            cache_dir: Directory for caching models (default: ~/.cache/huggingface)
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized HuggingFace model manager with cache: {self.cache_dir}")

    def search_models(
        self,
        query: str,
        task: Optional[str] = None,
        limit: int = 10,
    ) -> List[ModelInfo]:
        """
        Search for models on HuggingFace Hub

        Args:
            query: Search query
            task: Filter by task (e.g., "text-classification", "text-generation")
            limit: Maximum number of results

        Returns:
            List of model information
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            models = api.list_models(
                search=query,
                task=task,
                limit=limit,
                sort="downloads",
                direction=-1,
            )

            results = []
            for model in models:
                results.append(ModelInfo(
                    model_id=model.modelId,
                    task=getattr(model, 'pipeline_tag', 'unknown'),
                    description=getattr(model, 'description', ''),
                    downloads=getattr(model, 'downloads', 0),
                    likes=getattr(model, 'likes', 0),
                    tags=list(getattr(model, 'tags', [])),
                ))

            logger.info(f"Found {len(results)} models for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error searching models: {e}")
            # Return some common models as fallback
            return self._get_fallback_models(query, task)

    def _get_fallback_models(self, query: str, task: Optional[str] = None) -> List[ModelInfo]:
        """Get fallback models when API search fails"""

        common_models = {
            "text-classification": [
                ("distilbert-base-uncased-finetuned-sst-2-english", "Sentiment analysis"),
                ("bert-base-uncased", "General text classification"),
            ],
            "text-generation": [
                ("gpt2", "Small GPT-2 model"),
                ("distilgpt2", "Distilled GPT-2"),
            ],
            "question-answering": [
                ("distilbert-base-cased-distilled-squad", "Question answering on SQuAD"),
                ("bert-large-uncased-whole-word-masking-finetuned-squad", "BERT QA"),
            ],
            "token-classification": [
                ("dslim/bert-base-NER", "Named entity recognition"),
                ("dbmdz/bert-large-cased-finetuned-conll03-english", "NER on CoNLL"),
            ],
        }

        if task and task in common_models:
            return [
                ModelInfo(model_id=mid, task=task, description=desc)
                for mid, desc in common_models[task]
            ]

        # Return general purpose models
        return [
            ModelInfo(
                model_id="bert-base-uncased",
                task="fill-mask",
                description="Base BERT model",
            ),
            ModelInfo(
                model_id="distilbert-base-uncased",
                task="fill-mask",
                description="Distilled BERT",
            ),
        ]

    def load_model(
        self,
        config: ModelLoadConfig,
    ) -> Dict[str, Any]:
        """
        Load a HuggingFace model

        Args:
            config: Model loading configuration

        Returns:
            Dictionary with 'model', 'tokenizer', and 'config'
        """
        cache_key = f"{config.model_id}:{config.task}:{config.quantization}"

        # Return cached model if available
        if cache_key in self.loaded_models:
            logger.info(f"Using cached model: {config.model_id}")
            return self.loaded_models[cache_key]

        logger.info(f"Loading model: {config.model_id}")

        try:
            # Determine device
            if config.device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = config.device

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                use_fast=config.use_fast_tokenizer,
                cache_dir=config.cache_dir or self.cache_dir,
                trust_remote_code=config.trust_remote_code,
            )

            # Load config
            model_config = AutoConfig.from_pretrained(
                config.model_id,
                cache_dir=config.cache_dir or self.cache_dir,
                trust_remote_code=config.trust_remote_code,
            )

            # Load model based on task
            load_kwargs = {
                "cache_dir": config.cache_dir or self.cache_dir,
                "trust_remote_code": config.trust_remote_code,
            }

            # Add quantization if specified
            if config.quantization == "8bit":
                load_kwargs["load_in_8bit"] = True
            elif config.quantization == "4bit":
                load_kwargs["load_in_4bit"] = True

            # Select appropriate model class
            if config.task == "text-generation" or "gpt" in config.model_id.lower():
                model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
            elif config.task == "text-classification":
                model = AutoModelForSequenceClassification.from_pretrained(config.model_id, **load_kwargs)
            elif config.task == "token-classification":
                model = AutoModelForTokenClassification.from_pretrained(config.model_id, **load_kwargs)
            elif config.task == "question-answering":
                model = AutoModelForQuestionAnswering.from_pretrained(config.model_id, **load_kwargs)
            else:
                model = AutoModel.from_pretrained(config.model_id, **load_kwargs)

            # Move to device (if not quantized)
            if config.quantization is None:
                model = model.to(device)

            result = {
                "model": model,
                "tokenizer": tokenizer,
                "config": model_config,
                "device": device,
                "model_id": config.model_id,
                "task": config.task,
            }

            # Cache the model
            self.loaded_models[cache_key] = result

            logger.info(f"Successfully loaded model on {device}: {config.model_id}")
            return result

        except Exception as e:
            logger.error(f"Error loading model {config.model_id}: {e}")
            raise

    def create_pipeline(
        self,
        task: str,
        model_id: str,
        device: Optional[int] = None,
    ) -> Any:
        """
        Create a HuggingFace pipeline for easy inference

        Args:
            task: Pipeline task (e.g., "text-classification", "text-generation")
            model_id: Model identifier
            device: Device index (None for CPU, 0 for GPU 0, etc.)

        Returns:
            HuggingFace pipeline object
        """
        try:
            logger.info(f"Creating pipeline: {task} with model {model_id}")

            pipe = pipeline(
                task=task,
                model=model_id,
                device=device,
                cache_dir=self.cache_dir,
            )

            return pipe

        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise

    def fine_tune(
        self,
        model_id: str,
        train_dataset: Any,
        eval_dataset: Optional[Any],
        config: FineTuneConfig,
        compute_metrics: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune a HuggingFace model

        Args:
            model_id: Model to fine-tune
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config: Fine-tuning configuration
            compute_metrics: Function to compute metrics

        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting fine-tuning: {model_id}")

        try:
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_id, cache_dir=self.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=str(config.output_dir),
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=config.per_device_eval_batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                warmup_steps=config.warmup_steps,
                logging_steps=config.logging_steps,
                eval_steps=config.eval_steps,
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
                load_best_model_at_end=config.load_best_model_at_end,
                metric_for_best_model=config.metric_for_best_model,
                greater_is_better=config.greater_is_better,
                fp16=config.fp16,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                evaluation_strategy="steps" if eval_dataset else "no",
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            # Train
            logger.info("Starting training...")
            train_result = trainer.train()

            # Save model
            trainer.save_model()

            # Evaluate
            if eval_dataset:
                logger.info("Evaluating model...")
                eval_result = trainer.evaluate()
            else:
                eval_result = {}

            logger.info(f"Fine-tuning completed: {model_id}")

            return {
                "train_result": train_result,
                "eval_result": eval_result,
                "model_path": config.output_dir,
            }

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise

    def get_model_code_template(
        self,
        model_id: str,
        task: str,
        use_pipeline: bool = True,
    ) -> str:
        """
        Generate code template for using a HuggingFace model

        Args:
            model_id: Model identifier
            task: Task type
            use_pipeline: Whether to use pipeline API (simpler) or raw model

        Returns:
            Python code template as string
        """
        if use_pipeline:
            return f'''
from transformers import pipeline

# Create pipeline for {task}
pipe = pipeline("{task}", model="{model_id}")

# Example usage
result = pipe("Your input text here")
print(result)
'''
        else:
            return f'''
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModel.from_pretrained("{model_id}")

# Prepare input
inputs = tokenizer("Your input text here", return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)

# Process outputs
# ... add task-specific processing here
'''

    def unload_model(self, model_id: str, task: Optional[str] = None) -> None:
        """
        Unload a model from memory

        Args:
            model_id: Model identifier
            task: Task type (optional)
        """
        keys_to_remove = []
        for key in self.loaded_models:
            if key.startswith(f"{model_id}:"):
                if task is None or task in key:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.loaded_models[key]
            logger.info(f"Unloaded model: {key}")

    def clear_cache(self) -> None:
        """Clear all loaded models from memory"""
        self.loaded_models.clear()
        logger.info("Cleared model cache")
