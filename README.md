# SciAgent

**Advanced Multi-Agent Scientific & Software Development System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SciAgent is a production-ready system that combines cutting-edge AI techniques for scientific research and software development. It orchestrates multiple specialized AI agents (Gemini, Claude, GPT-4) with advanced techniques like Reflexion, extended thinking, and multi-agent debate to autonomously plan, execute, and analyze experiments.

## 🌟 Key Features

### Multi-Agent Architecture
- **Science Agent (Gemini)**: Literature search, hypothesis generation, experimental design, result analysis
- **Coding Agent (Claude)**: Code generation with Reflexion-based self-improvement and debugging
- **Data Agent**: Automated dataset downloading and preprocessing with HuggingFace integration
- **Overseer Agent (GPT-4)**: Multi-perspective validation and quality assurance
- **RAG Agent**: Retrieval-augmented generation with document and graph-based search

### Advanced AI Techniques
- **Extended Thinking**: Deep reasoning for complex scientific questions
- **Reflexion**: Self-critique and iterative code improvement
- **RAG & Graph RAG**: Production-ready retrieval with FAISS/ChromaDB and knowledge graphs
- **MCTS Planning**: Monte Carlo Tree Search for optimal experiment sequences
- **Bayesian Design**: Expected Information Gain for experiment selection
- **Thompson Sampling**: Efficient hypothesis space exploration
- **Knowledge Graph**: NetworkX-based scientific knowledge management
- **MCP Integration**: Model Context Protocol for literature search and tools
- **Constitutional AI**: Validation against scientific principles

### Human-in-the-Loop
- **Interactive Mode**: Approve each step of the experiment
- **Pause/Resume**: Save progress and continue later
- **Custom Callbacks**: Integrate your own approval logic
- **Progress Monitoring**: Real-time updates on experiment status

### Production-Ready
- **Error Handling**: Robust error recovery with agent-assisted debugging
- **Resource Management**: Timeout enforcement and monitoring
- **Modular Design**: Easily extend with custom agents and tools
- **Rich CLI**: Beautiful command-line interface with progress bars

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sciagent.git
cd sciagent

# Install with pip
pip install -e .

# Or install from PyPI (when published)
pip install sciagent
```

### Setup API Keys

Create a `.env` file in your project directory:

```bash
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

Or set environment variables:

```bash
export ANTHROPIC_API_KEY=your_anthropic_key
export GEMINI_API_KEY=your_gemini_key
export OPENAI_API_KEY=your_openai_key
```

### Basic Usage

#### Command Line

```bash
# Run a scientific experiment
sciagent run "Test if dropout improves model generalization on CIFAR-10"

# Automated mode (no approvals)
sciagent run "Compare SGD vs Adam on MNIST" --auto

# List experiments
sciagent list

# Pause an experiment
sciagent pause exp_abc123

# Resume an experiment
sciagent resume exp_abc123

# Initialize configuration
sciagent init-config
```

#### Python API

```python
import asyncio
from sciagent import SciAgent

async def main():
    # Create agent
    agent = SciAgent()

    # Run experiment
    result = await agent.run(
        "Test if dropout improves generalization on CIFAR-10",
        interactive=False  # Automated mode
    )

    # Access results
    print(result.analysis.summary)
    print(f"Confidence: {result.analysis.confidence:.2%}")
    print(f"Code Quality: {result.code.quality_score:.2%}")

asyncio.run(main())
```

#### Interactive Mode with Custom Callbacks

```python
import asyncio
from sciagent import SciAgent
from sciagent.utils.models import HumanApprovalRequest

async def my_approval_handler(request: HumanApprovalRequest):
    """Custom approval logic"""

    if request.request_type == "approval":
        # Auto-approve data preparation
        if "data" in request.context.get("message", "").lower():
            request.response.set_result("yes")
        else:
            # Ask user for other approvals
            user_input = input(f"{request.context['message']} (yes/no): ")
            request.response.set_result(user_input)

    elif request.request_type == "choice":
        # Display options and get user choice
        for i, opt in enumerate(request.options):
            print(f"{i+1}. {opt}")
        choice = int(input("Select: ")) - 1
        request.response.set_result(request.options[choice])

async def main():
    agent = SciAgent()

    result = await agent.run(
        "Your research question",
        interactive=True,
        on_approval=my_approval_handler
    )

asyncio.run(main())
```

## 📖 Examples

### Scientific Research

```python
# Hypothesis testing
result = await agent.run("""
Test if batch normalization improves training stability on ResNet-50:
- Use ImageNet dataset
- Compare with/without batch norm
- Measure convergence rate and final accuracy
- Run 5-fold cross-validation
""")
```

### Software Development

```python
# Build a web API
result = await agent.run("""
Build a FastAPI endpoint for user authentication with:
- JWT tokens
- Password hashing
- Rate limiting
- Input validation
- Unit tests
""")

# Save generated code
with open("auth_api.py", "w") as f:
    f.write(result.code.code)
```

### Data Analysis

```python
# Automated data analysis
result = await agent.run("""
Analyze the relationship between study hours and exam scores:
- Load data from student_data.csv
- Perform correlation analysis
- Create visualizations
- Test for statistical significance
- Generate report with findings
""")
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
│           CLI, Python API, Interactive Mode                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               Coordinator (Orchestrator)                     │
│  • Human-in-the-Loop Manager                                 │
│  • State Machine (pause/resume)                              │
│  • Process Manager                                           │
└─────┬───────────────────────────────────────────────────────┘
      │
      ├──────────────┬──────────────┬──────────────┬──────────
      ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Science  │  │ Coding   │  │ Data     │  │ Overseer │
│ Agent    │  │ Agent    │  │ Agent    │  │ Agent    │
│(Gemini)  │  │(Claude)  │  │(Auto)    │  │(GPT-4)   │
│          │  │          │  │          │  │          │
│Extended  │  │Reflexion │  │Dataset   │  │Multi-    │
│Thinking  │  │Self-Fix  │  │Download  │  │Agent     │
│          │  │          │  │Loaders   │  │Debate    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

## 🎯 Workflow

1. **Scientific Planning** (Science Agent)
   - Searches literature
   - Generates hypotheses
   - Designs experiments
   - Proposes measurement strategies

2. **Data Preparation** (Data Agent)
   - Downloads datasets automatically
   - Creates data loaders
   - Prepares preprocessing pipelines

3. **Code Generation** (Coding Agent)
   - Generates experiment code
   - Self-critiques and improves (Reflexion)
   - Adds tests and documentation

4. **Execution** (Monitored Executor)
   - Runs code with safety checks
   - Real-time monitoring
   - Error handling and recovery

5. **Analysis** (Science Agent)
   - Analyzes results
   - Statistical testing
   - Interprets findings

6. **Validation** (Overseer Agent)
   - Constitutional AI review
   - Multi-agent debate
   - Quality assurance

7. **Refinement** (All Agents)
   - Proposes improvements
   - Suggests follow-up experiments
   - Iterative optimization

## ⚙️ Configuration

Create a `~/.sciagent/config.yaml`:

```yaml
agents:
  science:
    model: "gemini-2.0-flash-thinking-exp"
    temperature: 0.7

  coding:
    model: "claude-sonnet-4-20250514"
    temperature: 0.7

  overseer:
    model: "gpt-4"
    temperature: 0.3

features:
  enable_extended_thinking: true
  enable_reflexion: true
  enable_debate: true

execution:
  max_concurrent_experiments: 5
  default_timeout: 3600

logging:
  log_level: "INFO"
```

## 📊 Supported Datasets

SciAgent includes built-in handlers for:

- **Computer Vision**: CIFAR-10, CIFAR-100, MNIST, ImageNet, COCO
- **NLP**: Coming soon
- **Custom**: Automatically generates loaders for custom datasets

## 🔧 Advanced Features

### Reflexion (Self-Improving Code)

The Coding Agent uses Reflexion to iteratively improve code quality:

```python
# Enable Reflexion (default)
config.enable_reflexion = True
config.reflexion_max_iterations = 5
config.reflexion_quality_threshold = 0.95
```

### Extended Thinking

Science Agent uses extended thinking for complex reasoning:

```python
# Enable extended thinking (default)
config.enable_extended_thinking = True
```

### Constitutional AI

Experiments are validated against scientific principles:

```python
# Principles checked:
# - Reproducibility
# - Transparency
# - Statistical rigor
# - Ethical data use
# - Bias awareness
# - Peer review
```

### Multi-Agent Debate

Multiple agents debate to reach consensus:

```python
config.enable_debate = True
config.debate_rounds = 3
config.debate_consensus_threshold = 0.75
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=sciagent --cov-report=html
```

## 📝 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/

# Lint
ruff check src/

# Type check
mypy src/
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🚀 Advanced Features

### RAG (Retrieval-Augmented Generation)

Production-ready RAG with multiple backends and strategies:

#### Traditional RAG

```python
from sciagent.rag import RAGPipeline

# Initialize with FAISS (fast) or ChromaDB (persistent)
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_type="faiss",  # or "chroma"
    chunk_size=512,
    chunk_overlap=50,
)

# Add documents
documents = [
    "Dropout prevents overfitting in neural networks...",
    "Batch normalization improves training stability...",
]
rag.add_documents(documents)

# Query with automatic re-ranking
results, context = rag.query("How to prevent overfitting?", top_k=5)

# Use context with LLM
prompt = f"Question: How to prevent overfitting?\n\nContext:\n{context}\n\nAnswer:"
```

#### Graph RAG

Combine knowledge graphs with RAG for relationship-aware retrieval:

```python
from sciagent.rag import GraphRAG
from sciagent.project.knowledge_graph import ScientificKnowledgeGraph

# Initialize with knowledge graph
kg = ScientificKnowledgeGraph()
graph_rag = GraphRAG(knowledge_graph=kg, max_hops=3)

# Query with multi-hop reasoning
results, context = graph_rag.query(
    "What papers discuss transformer attention mechanisms?",
    top_k=5,
    use_multi_hop=True  # Traverse graph for complex queries
)

# Results include relationship context
for result in results:
    print(f"{result.node.node_type}: {result.node.content}")
    print(f"Related nodes: {len(result.related_nodes)}")
    print(f"Paths from query: {len(result.paths)}")
```

#### Hybrid RAG

Combine traditional and graph RAG for best results:

```python
from sciagent.rag import HybridRAG

hybrid = HybridRAG(
    rag_pipeline=rag,
    graph_rag=graph_rag,
    combination_strategy="weighted"
)

# Weighted combination of both approaches
results, context = hybrid.query(
    query="Your question",
    top_k=5,
    rag_weight=0.6  # 60% traditional RAG, 40% graph RAG
)
```

#### RAG Agent

Easy integration with SciAgent workflows:

```python
from sciagent.agents import RAGAgent

rag_agent = RAGAgent(config)

# Add documents
await rag_agent.add_documents([
    "Document 1...",
    "Document 2...",
])

# Query with different modes
result = await rag_agent.query(
    query="Your question",
    mode="hybrid",  # "rag", "graph", or "hybrid"
    top_k=5
)

print(result["context"])
```

**Features**:
- **Vector Stores**: FAISS (fast), ChromaDB (persistent), or in-memory
- **Embedding Models**: Sentence Transformers, OpenAI, or custom
- **Chunking**: Configurable size and overlap
- **Re-ranking**: Improved result quality
- **Hybrid Search**: Semantic + keyword + graph structure
- **Multi-hop**: Graph traversal for complex queries
- **Production Ready**: Persistent, scalable, monitored

See [docs/RAG_GUIDE.md](docs/RAG_GUIDE.md) for complete guide.

### Knowledge Graph

Store and query scientific knowledge:

```python
from sciagent.project.knowledge_graph import ScientificKnowledgeGraph

kg = ScientificKnowledgeGraph()

# Query related papers
papers = kg.query_related_papers("dropout regularization", limit=10)

# Get statistics
stats = kg.get_statistics()

# Visualize
kg.visualize("knowledge_graph.png")

# Export/Import
kg.export_to_json("kg_export.json")
```

### MCTS Experiment Planning

Plan optimal sequences of experiments:

```python
from sciagent.advanced import MCTSPlanner

planner = MCTSPlanner(n_simulations=100, max_depth=5)

sequence = await planner.plan(
    hypothesis=hypothesis,
    initial_design=design,
    budget=5000
)

print(f"Planned {len(sequence.steps)} experiments")
print(f"Expected value: {sequence.total_value:.2f}")
```

### Bayesian Experimental Design

Select experiments that maximize information gain:

```python
from sciagent.advanced import BayesianExperimentSelector, ExperimentCandidate

selector = BayesianExperimentSelector()

candidates = [
    ExperimentCandidate(
        description="Increase sample size",
        parameters={"n": 2000},
        uncertainty=0.8,
        expected_cost=500.0
    ),
    # ... more candidates
]

selected = await selector.select_experiments(
    candidates=candidates,
    prior_results=[],
    max_select=3
)
```

### Thompson Sampling

Efficiently explore hypothesis space:

```python
from sciagent.advanced import ThompsonSamplingExplorer

explorer = ThompsonSamplingExplorer()

hypotheses = await explorer.explore(
    query="attention mechanisms",
    papers=papers,
    n_iterations=20
)

# Get best hypotheses
best = explorer.get_best_hypotheses(n=5)
```

### MCP Integration

Access external tools and knowledge:

```python
from sciagent.mcp import create_default_mcp_client

mcp_client = create_default_mcp_client()
await mcp_client.initialize()

# Search arXiv
papers = await mcp_client.call_tool(
    server="arxiv",
    tool="search_papers",
    arguments={"query": "transformer attention", "limit": 10}
)
```

## 🤗 HuggingFace Integration

SciAgent provides seamless integration with HuggingFace's ecosystem:

### Models

Access 100,000+ pre-trained models from HuggingFace Hub:

```python
from sciagent.integrations.huggingface import HuggingFaceModelManager

manager = HuggingFaceModelManager()

# Search for models
models = manager.search_models("sentiment", task="text-classification", limit=5)

# Create pipeline for inference
pipe = manager.create_pipeline(
    task="text-classification",
    model_id="distilbert-base-uncased-finetuned-sst-2-english"
)

result = pipe("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Load model manually for fine-tuning
from sciagent.integrations.huggingface.models import ModelLoadConfig

config = ModelLoadConfig(
    model_id="bert-base-uncased",
    task="text-classification",
    quantization="8bit",  # Optional: reduce memory
)

model_dict = manager.load_model(config)
```

### Datasets

Access 50,000+ datasets from HuggingFace Datasets:

```python
from sciagent.integrations.huggingface import HuggingFaceDatasetManager
from sciagent.integrations.huggingface.datasets import DatasetLoadConfig

manager = HuggingFaceDatasetManager()

# Search for datasets
datasets = manager.search_datasets("sentiment", task="text-classification", limit=5)

# Load dataset
config = DatasetLoadConfig(dataset_id="imdb", split="train")
dataset = manager.load_dataset(config)

print(f"Loaded {len(dataset)} samples")
# Loaded 25000 samples

# Tokenize dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized = manager.tokenize_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    text_column="text",
    max_length=512
)

# Create DataLoader
from sciagent.integrations.huggingface.datasets import DataLoaderConfig

dataloader = manager.create_dataloader(
    dataset=tokenized,
    config=DataLoaderConfig(batch_size=32, shuffle=True)
)
```

### Integration with SciAgent

Use HuggingFace datasets directly in experiments:

```python
from sciagent import SciAgent

agent = SciAgent()

# Use "hf:" prefix for HuggingFace datasets
result = await agent.run("""
Test if fine-tuning BERT improves sentiment classification:
- Use HuggingFace IMDB dataset (hf:imdb)
- Fine-tune bert-base-uncased
- Compare with baseline
- Measure accuracy and F1 score
""", interactive=False)
```

**Automatic Model Suggestions:**

SciAgent automatically suggests appropriate HuggingFace models based on your task:

- **Text Classification**: distilbert, bert-base-uncased, roberta
- **Question Answering**: distilbert-squad, bert-large-squad
- **Summarization**: bart-large-cnn, t5-base
- **Translation**: t5-base, opus-mt models
- **Image Classification**: vit-base, resnet-50
- **Vision-Language**: clip-vit-base

**Features:**
- Search and discover models and datasets
- Pipeline API for easy inference
- Manual loading for fine-tuning
- Tokenization and preprocessing utilities
- PyTorch DataLoader creation
- Memory optimization with quantization
- Streaming support for large datasets
- Code template generation

See [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) for complete guide and examples.

## 💻 Cursor Integration

SciAgent works seamlessly with Cursor IDE and Claude Code:

### Setup

1. Open project in Cursor
2. MCP servers are auto-configured via `.cursor/mcp_settings.json`
3. Use `.cursorrules` for Claude Code context

### Usage

Ask Claude Code in Cursor:

```
"Create an experiment that uses Thompson sampling to explore
hypotheses about batch normalization, then uses MCTS to plan
the optimal sequence of experiments"
```

Claude Code will:
- Understand SciAgent architecture
- Generate code using appropriate components
- Follow project patterns

See [docs/CURSOR_SETUP.md](docs/CURSOR_SETUP.md) for detailed guide.

## 🙏 Acknowledgments

SciAgent builds on cutting-edge research:

- **Reflexion**: Self-Improvement through Verbal Reinforcement Learning
- **Constitutional AI**: Training Language Models to Follow Principles
- **Extended Thinking**: Gemini's Deep Reasoning Capabilities
- **MCTS**: Monte Carlo Tree Search for Planning
- **Bayesian Optimization**: Expected Information Gain for Experimental Design
- **Thompson Sampling**: Bayesian Bandit Algorithms
- **Model Context Protocol**: Anthropic's MCP for tool integration

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/sciagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sciagent/discussions)

## 🗺️ Roadmap

- [x] ✅ MCTS planning
- [x] ✅ Bayesian experimental design
- [x] ✅ Thompson sampling
- [x] ✅ Knowledge graph
- [x] ✅ MCP integration
- [x] ✅ Cursor integration
- [ ] Web interface
- [ ] More dataset handlers
- [ ] FunSearch evolution (in progress)
- [ ] Multi-modal support
- [ ] Cloud deployment
- [ ] Collaboration features
- [ ] RAG integration for documents

## ⭐ Star History

If you find SciAgent useful, please star the repository!

---

**Built with ❤️ for the scientific community**
