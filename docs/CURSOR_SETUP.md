# Cursor Setup Guide for SciAgent

This guide explains how to use SciAgent effectively within the Cursor IDE with Claude Code integration.

## Prerequisites

1. **Cursor IDE** installed (https://cursor.sh/)
2. **Claude Code** extension enabled in Cursor
3. **Node.js** (for MCP servers)
4. **Python 3.10+** with virtual environment

## Initial Setup

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/sciagent.git
cd sciagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install SciAgent in development mode
pip install -e ".[dev]"
```

### 2. Configure API Keys

Create `.env` file in the project root:

```bash
# Copy example
cp .env.example .env

# Edit .env and add your API keys
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional: for MCP servers
GITHUB_TOKEN=your_github_token_here
BRAVE_API_KEY=your_brave_api_key_here
```

### 3. Configure MCP Servers

SciAgent uses Model Context Protocol for enhanced capabilities:

#### Option A: Automatic (Recommended)

The `.cursor/mcp_settings.json` file is already configured. Cursor will automatically detect and use these MCP servers.

#### Option B: Manual Configuration

If automatic detection doesn't work:

1. Open Cursor Settings
2. Navigate to "Model Context Protocol"
3. Add servers manually:

```json
{
  "arxiv": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-arxiv"]
  },
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
  }
}
```

### 4. Initialize SciAgent Configuration

```bash
# Create default configuration
sciagent init-config

# This creates ~/.sciagent/config.yaml
# Edit as needed for your preferences
```

## Using SciAgent in Cursor

### Quick Start

1. **Open project in Cursor**
   ```bash
   cursor .
   ```

2. **Activate Claude Code**
   - Press `Cmd+K` (Mac) or `Ctrl+K` (Windows/Linux)
   - Claude Code will activate

3. **Run an experiment**
   ```bash
   # Terminal in Cursor
   sciagent run "Test dropout on CIFAR-10"
   ```

### Workflow Integration

#### 1. Interactive Development

Use Claude Code to:

- **Generate experiments**: "Create an experiment to test batch normalization effectiveness"
- **Debug code**: "Why is the Reflexion loop not converging?"
- **Explain components**: "How does the MCTS planner work?"
- **Modify agents**: "Add a new MCP server for PubMed search"

#### 2. Code Generation

Ask Claude Code to generate experiment code:

```
"Generate a SciAgent experiment that:
1. Searches papers on transformer attention mechanisms
2. Generates 5 hypotheses about attention head specialization
3. Tests the top hypothesis with a BERT model
4. Uses MCTS to plan optimal experiment sequence"
```

Claude Code will:
- Understand SciAgent architecture (from `.cursorrules`)
- Generate code following project patterns
- Use appropriate agents and components

#### 3. Knowledge Graph Queries

Use Python API in Cursor terminal:

```python
from sciagent.project.knowledge_graph import ScientificKnowledgeGraph

# Load knowledge graph
kg = ScientificKnowledgeGraph()

# Query related papers
papers = kg.query_related_papers("dropout regularization")

# Get statistics
stats = kg.get_statistics()
print(stats)

# Visualize
kg.visualize("knowledge_graph.png")
```

### MCP Integration Examples

#### Search Literature During Development

In Cursor, ask Claude Code:

```
"Search arXiv for recent papers on self-attention mechanisms
and summarize the key findings"
```

Claude Code will use the arxiv MCP server to fetch real papers.

#### Code Search

```
"Find examples of MCTS implementation in GitHub repositories
similar to our MCTSPlanner"
```

Claude Code uses github MCP server for code search.

### Advanced Features

#### 1. Bayesian Experimental Design

```python
from sciagent.advanced import BayesianExperimentSelector, ExperimentCandidate

selector = BayesianExperimentSelector()

candidates = [
    ExperimentCandidate(
        description="Test with larger learning rate",
        parameters={"lr": 0.01},
        uncertainty=0.8,
        expected_cost=100.0
    ),
    # ... more candidates
]

selected = await selector.select_experiments(
    candidates=candidates,
    prior_results=[],
    max_select=3
)
```

#### 2. MCTS Planning

```python
from sciagent.advanced import MCTSPlanner
from sciagent.utils.models import Hypothesis, ExperimentDesign

planner = MCTSPlanner(n_simulations=100)

hypothesis = Hypothesis(
    statement="Dropout improves generalization",
    rationale="Prevents overfitting",
    testable_predictions=["Better val accuracy"],
    expected_evidence="Validation metrics"
)

design = ExperimentDesign(
    description="Test dropout rates",
    measurements=["accuracy", "loss"],
    controls=["no_dropout"],
    sample_size=1000,
    statistical_tests=["t-test"],
    potential_confounds=["batch_size"],
    success_criteria="p < 0.05"
)

sequence = await planner.plan(
    hypothesis=hypothesis,
    initial_design=design,
    budget=5000
)

print(f"Planned {len(sequence.steps)} experiments")
print(f"Expected value: {sequence.total_value:.2f}")
```

#### 3. Thompson Sampling for Hypotheses

```python
from sciagent.advanced import ThompsonSamplingExplorer

explorer = ThompsonSamplingExplorer()

hypotheses = await explorer.explore(
    query="impact of attention heads",
    papers=papers,
    n_iterations=20
)

# Get best hypotheses
best = explorer.get_best_hypotheses(n=3)
for h in best:
    print(f"{h.hypothesis.statement} - EV: {h.expected_value:.3f}")
```

## Cursor-Specific Tips

### 1. Use `.cursorrules` Effectively

The `.cursorrules` file teaches Claude Code about:
- Project architecture
- Code patterns
- Development guidelines
- Common tasks

Ask Claude Code questions like:
- "How do I add a new agent?"
- "What's the pattern for MCP integration?"
- "Show me how to implement a new advanced technique"

### 2. Inline Code Generation

Highlight a section of code and use `Cmd+K`:

```
"Add error handling and logging to this function following SciAgent patterns"
```

Claude Code will add appropriate try/except blocks and logging.

### 3. Multi-File Editing

Ask Claude Code to make changes across multiple files:

```
"Add support for a new dataset 'FashionMNIST' by:
1. Creating a handler in data_agent.py
2. Adding tests in test_data_agent.py
3. Adding an example in examples/
4. Updating the README"
```

### 4. Debugging with Context

When encountering errors, paste the traceback and ask:

```
"This error occurred during MCTS planning.
Debug and fix, ensuring it follows SciAgent error handling patterns"
```

## Best Practices

### 1. Keep Knowledge Graph Updated

```python
# After experiments, update knowledge graph
kg.update(hypothesis, results, analysis)
kg._save_to_cache()
```

### 2. Use Async Properly

Always use `async def` for agent methods and `await` for calls:

```python
# Good
result = await agent.process(request)

# Bad (will return coroutine, not result)
result = agent.process(request)
```

### 3. Log Appropriately

```python
# High-level operations
logger.info("Starting experiment planning")

# Detailed progress
logger.debug(f"Testing hypothesis: {hypothesis.statement}")

# Errors
logger.error(f"Failed to load model: {e}")
```

### 4. Handle User Approvals

Test both interactive and automated modes:

```python
# Interactive - asks user
result = await agent.run(query, interactive=True)

# Automated - no approvals
result = await agent.run(query, interactive=False)
```

## Troubleshooting

### MCP Servers Not Working

1. Check Node.js is installed: `node --version`
2. Verify MCP settings in Cursor
3. Check logs: Cursor > Settings > Model Context Protocol > View Logs

### API Keys Not Found

1. Verify `.env` file exists in project root
2. Check keys are set: `cat .env | grep API_KEY`
3. Reload Cursor to pick up changes

### Import Errors

```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import sciagent; print(sciagent.__version__)"
```

### Async Errors

Make sure to use `asyncio.run()` in scripts:

```python
import asyncio

async def main():
    result = await agent.run(query)

if __name__ == "__main__":
    asyncio.run(main())
```

## Example Workflows

### 1. Literature Review + Experiment

```python
from sciagent import SciAgent

agent = SciAgent()

# This will:
# 1. Search arXiv via MCP
# 2. Build knowledge graph
# 3. Generate hypotheses with Thompson sampling
# 4. Plan experiments with MCTS
# 5. Execute and analyze

result = await agent.run(
    "Investigate the role of residual connections in deep networks",
    interactive=True
)
```

### 2. Iterative Refinement

```python
# Initial experiment
result1 = await agent.run("Test dropout on MNIST")

# Examine refinements
for i, ref in enumerate(result1.refinements):
    print(f"{i+1}. {ref.description} (EIG: {ref.expected_information_gain:.2f})")

# Run selected refinement
result2 = await agent.run(
    f"MNIST dropout - {result1.refinements[0].description}",
    interactive=False
)
```

### 3. Multi-Hypothesis Testing

```python
from sciagent.advanced import ThompsonSamplingExplorer

explorer = ThompsonSamplingExplorer()

# Explore multiple hypotheses
hypotheses = await explorer.explore(
    query="batch normalization effects",
    papers=papers,
    n_iterations=50
)

# Test top 3
for h_stats in hypotheses[:3]:
    result = await agent.run(
        f"Test hypothesis: {h_stats.hypothesis.statement}",
        interactive=False
    )

    # Update explorer with results
    reward = 1.0 if result.validation.approved else 0.0
    explorer._update(h_stats, reward)
```

## Additional Resources

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **API Reference**: Run `sciagent --help`
- **Knowledge Graph Visualization**: Use `kg.visualize("graph.png")`

## Support

For issues or questions:
1. Check documentation in `docs/`
2. Search existing issues on GitHub
3. Ask Claude Code in Cursor (it has full project context!)
4. Open a new issue with details

---

**Happy experimenting with SciAgent in Cursor! 🔬**
