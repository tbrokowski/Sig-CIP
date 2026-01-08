"""
Coding Agent (Claude) with Reflexion & Self-Improvement

Handles code generation, debugging, and optimization with self-critique loops.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

try:
    import anthropic
except ImportError:
    anthropic = None

from sciagent.agents.base import BaseAgent
from sciagent.utils.config import Config
from sciagent.utils.models import CodeArtifact, Critique, TestResult


class CodingAgent(BaseAgent):
    """
    Claude-powered coding agent with:
    - Reflexion for self-improvement
    - Project-aware code generation
    - Automated debugging
    """

    def __init__(self, config: Config):
        super().__init__(config)

        if anthropic is None:
            raise ImportError(
                "anthropic not installed. Install with: pip install anthropic"
            )

        # Initialize Claude client
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = config.coding_agent.model or "claude-sonnet-4-20250514"

        # Reflexion engine (simplified)
        self.reflexion_enabled = config.enable_reflexion
        self.max_iterations = config.reflexion_max_iterations

    async def process(self, request: Dict[str, Any]) -> Any:
        """
        Process a request

        Supported actions:
        - implement_experiment: Generate experiment code
        - debug_code: Debug failed code
        """

        action = request.get("action")

        if action == "implement_experiment":
            return await self.implement_experiment(
                request["plan"], request.get("data_info")
            )
        elif action == "debug_code":
            return await self.debug_code(
                request["code"], request["error"], request["traceback"]
            )
        else:
            raise ValueError(f"Unknown action: {action}")

    async def implement_experiment(
        self, plan: Any, data_info: Any = None
    ) -> CodeArtifact:
        """
        Generate experiment code with self-improvement

        Args:
            plan: Experiment plan
            data_info: Optional data preparation info

        Returns:
            Code artifact with quality score
        """

        self.logger.info("Generating experiment code")

        # Build comprehensive prompt
        prompt = self._build_implementation_prompt(plan, data_info)

        # Use Reflexion for self-improving code generation if enabled
        if self.reflexion_enabled:
            code_artifact = await self._generate_with_reflection(
                prompt=prompt,
                max_iterations=self.max_iterations,
            )
        else:
            # Simple generation
            code = await self._generate_code(prompt)
            code_artifact = CodeArtifact(code=code, quality_score=0.7, iterations=1)

        return code_artifact

    async def debug_code(self, code: str, error: str, traceback: str) -> CodeArtifact:
        """
        Debug failed code with deep reflection

        Args:
            code: Failed code
            error: Error message
            traceback: Stack trace

        Returns:
            Fixed code artifact
        """

        self.logger.info(f"Debugging code - Error: {error}")

        # Generate reflection on error
        reflection = await self._deep_reflection(code, error, traceback)

        # Generate fix
        prompt = f"""This code failed:

```python
{code}
```

Error: {error}

Traceback:
{traceback}

Reflection on root cause:
{reflection}

Fix the code, addressing the root cause. Provide only the corrected code."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )

            fixed_code = self._extract_code(response.content[0].text)

            return CodeArtifact(
                code=fixed_code,
                quality_score=0.85,
                reflection=reflection,
            )

        except Exception as e:
            self.logger.error(f"Error during debugging: {e}")
            # Return original code with error noted
            return CodeArtifact(
                code=code,
                quality_score=0.3,
                failed=True,
                reflection=f"Debugging failed: {str(e)}",
            )

    async def _generate_with_reflection(
        self, prompt: str, max_iterations: int
    ) -> CodeArtifact:
        """
        Generate code with iterative self-critique (Reflexion)

        Args:
            prompt: Code generation prompt
            max_iterations: Maximum improvement iterations

        Returns:
            Best code artifact found
        """

        conversation = []
        best_artifact = None
        best_score = 0

        for iteration in range(max_iterations):
            self.logger.info(f"Reflexion iteration {iteration + 1}/{max_iterations}")

            # Generate code
            code = await self._generate_code(prompt, conversation)

            # Self-critique
            critique = await self._self_critique(code, iteration)

            current_score = critique.quality_score

            if current_score > best_score:
                best_artifact = CodeArtifact(
                    code=code,
                    quality_score=current_score,
                    iterations=iteration + 1,
                )
                best_score = current_score

            # Good enough?
            if current_score > self.config.reflexion_quality_threshold:
                self.logger.info(
                    f"Quality threshold reached: {current_score:.2f}"
                )
                return best_artifact

            # Continue improving
            conversation.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            conversation.append(
                {
                    "role": "user",
                    "content": f"Quality score: {current_score:.2f}\n\nFeedback:\n{critique.feedback}\n\nImprove the code based on this feedback.",
                }
            )

        return best_artifact or CodeArtifact(code=code, quality_score=0.5, iterations=max_iterations)

    async def _generate_code(self, prompt: str, conversation: list = None) -> str:
        """Generate code using Claude"""

        messages = conversation or []
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=self.config.coding_agent.temperature,
                messages=messages,
            )

            code = self._extract_code(response.content[0].text)
            return code

        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            # Return a minimal placeholder
            return f"# Error generating code: {str(e)}\npass"

    async def _self_critique(self, code: str, iteration: int) -> Critique:
        """
        Agent critiques its own code

        Args:
            code: Code to critique
            iteration: Current iteration number

        Returns:
            Critique with quality scores and feedback
        """

        critique_prompt = f"""Critique this code across multiple dimensions:

```python
{code}
```

Evaluate:
1. CORRECTNESS (0-1): Does it solve the problem correctly?
2. EFFICIENCY (0-1): Is it computationally efficient?
3. READABILITY (0-1): Is it clear and well-documented?
4. SCIENTIFIC_RIGOR (0-1): Are controls and statistical tests appropriate?
5. REPRODUCIBILITY (0-1): Can others reproduce this?

For each dimension, provide:
- Score (0.0 to 1.0)
- Brief explanation

Then provide specific improvement suggestions.

Format:
CORRECTNESS: [score] - [explanation]
EFFICIENCY: [score] - [explanation]
READABILITY: [score] - [explanation]
SCIENTIFIC_RIGOR: [score] - [explanation]
REPRODUCIBILITY: [score] - [explanation]

IMPROVEMENTS:
- [suggestion 1]
- [suggestion 2]
..."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": critique_prompt}],
            )

            critique_text = response.content[0].text

            # Parse scores (simplified)
            dimension_scores = self._parse_dimension_scores(critique_text)
            overall = sum(dimension_scores.values()) / len(dimension_scores)

            return Critique(
                quality_score=overall,
                dimension_scores=dimension_scores,
                feedback=critique_text,
            )

        except Exception as e:
            self.logger.error(f"Error during self-critique: {e}")
            # Return moderate scores as fallback
            return Critique(
                quality_score=0.7,
                dimension_scores={
                    "correctness": 0.7,
                    "efficiency": 0.7,
                    "readability": 0.7,
                    "scientific_rigor": 0.7,
                    "reproducibility": 0.7,
                },
                feedback="Critique failed - using default scores",
            )

    async def _deep_reflection(self, code: str, error: str, traceback: str) -> str:
        """
        Deep reflection on code failure

        Args:
            code: Failed code
            error: Error message
            traceback: Stack trace

        Returns:
            Reflection text
        """

        reflection_prompt = f"""Analyze why this code failed:

```python
{code}
```

Error: {error}
Traceback: {traceback}

Think deeply about:
1. What is the root cause of this error?
2. What assumptions were violated?
3. What should have been checked?
4. How can this be prevented in the future?

Provide a detailed analysis of the root cause."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": reflection_prompt}],
            )

            return response.content[0].text

        except Exception as e:
            return f"Reflection failed: {str(e)}"

    def _build_implementation_prompt(self, plan: Any, data_info: Any) -> str:
        """Build comprehensive implementation prompt"""

        prompt = f"""Generate Python code to implement this experiment:

Hypothesis: {plan.hypothesis.statement}

Experimental Design:
{plan.design.description}

Measurements needed: {', '.join(plan.design.measurements)}
Controls: {', '.join(plan.design.controls)}
Sample size: {plan.design.sample_size}
Statistical tests: {', '.join(plan.design.statistical_tests)}

"""

        if data_info:
            prompt += f"""
Dataset Information:
- Name: {data_info.dataset_name}
- Cache path: {data_info.cache_path}

Use this data loader code:
```python
{data_info.loader_code}
```
"""

        prompt += """
Requirements:
1. Import all necessary libraries
2. Set random seeds for reproducibility
3. Implement the experiment with proper controls
4. Include statistical tests
5. Generate plots for visualization
6. Print clear summary of results
7. Handle errors gracefully
8. Add comments explaining key steps

Provide complete, executable Python code."""

        return prompt

    def _extract_code(self, text: str) -> str:
        """Extract Python code from text"""

        # Look for code blocks
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()

        # If no code blocks, return whole text
        return text.strip()

    def _parse_dimension_scores(self, critique_text: str) -> Dict[str, float]:
        """Parse dimension scores from critique"""

        dimensions = {
            "correctness": 0.7,
            "efficiency": 0.7,
            "readability": 0.7,
            "scientific_rigor": 0.7,
            "reproducibility": 0.7,
        }

        # Simple parsing - look for "DIMENSION: score"
        for dim in dimensions.keys():
            key = dim.upper()
            if key in critique_text:
                # Try to extract score
                try:
                    line = [l for l in critique_text.split("\n") if key in l][0]
                    # Look for number between 0 and 1
                    import re

                    match = re.search(r"(\d+\.?\d*)", line)
                    if match:
                        score = float(match.group(1))
                        if score <= 1.0:
                            dimensions[dim] = score
                        elif score <= 10.0:  # Might be out of 10
                            dimensions[dim] = score / 10.0
                except:
                    pass

        return dimensions
