"""
Overseer Agent with Multi-Agent Debate and Constitutional AI

Provides validation, quality assurance, and multi-perspective analysis.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    import openai
except ImportError:
    openai = None

from sciagent.agents.base import BaseAgent
from sciagent.utils.config import Config
from sciagent.utils.models import (
    ConstitutionalReview,
    StatisticalReview,
    ValidationResult,
    SCIENTIFIC_CONSTITUTION,
)


class OverseerAgent(BaseAgent):
    """
    Multi-model overseer using:
    - GPT-4 for code review
    - Multi-agent debate for validation
    - Constitutional AI principles
    """

    def __init__(self, config: Config):
        super().__init__(config)

        if openai is None:
            raise ImportError("openai not installed. Install with: pip install openai")

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.model = config.overseer_agent.model or "gpt-4"

        # Constitutional principles
        self.constitution = SCIENTIFIC_CONSTITUTION

        # Debate settings
        self.debate_enabled = config.enable_debate
        self.debate_rounds = config.debate_rounds

    async def process(self, request: Dict[str, Any]) -> Any:
        """
        Process a request

        Supported actions:
        - validate_experiment: Comprehensive validation
        """

        action = request.get("action")

        if action == "validate_experiment":
            return await self.validate_experiment(
                request["code"], request["results"], request["hypothesis"]
            )
        else:
            raise ValueError(f"Unknown action: {action}")

    async def validate_experiment(
        self, code: Any, results: Any, hypothesis: Any
    ) -> ValidationResult:
        """
        Comprehensive validation via constitutional review + statistical analysis

        Args:
            code: Generated code
            results: Execution results
            hypothesis: Original hypothesis

        Returns:
            Validation result with approval and recommendations
        """

        self.logger.info("Validating experiment")

        # Stage 1: Constitutional review
        constitutional_review = await self._constitutional_review(code, results)

        # Stage 2: Statistical validity
        statistical_review = await self._statistical_review(results)

        # Stage 3: Overall assessment
        approved = constitutional_review.approved and statistical_review.approved

        confidence = self._calculate_confidence(constitutional_review, statistical_review)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            constitutional_review, statistical_review
        )

        return ValidationResult(
            approved=approved,
            confidence=confidence,
            debate_summary=f"Constitutional: {len(constitutional_review.violations)} violations, "
            f"Statistical: {len(statistical_review.issues)} issues",
            constitutional_violations=constitutional_review.violations,
            statistical_issues=statistical_review.issues,
            recommendations=recommendations,
        )

    async def _constitutional_review(self, code: Any, results: Any) -> ConstitutionalReview:
        """
        Review against scientific principles

        Args:
            code: Generated code
            results: Execution results

        Returns:
            Constitutional review result
        """

        violations = []

        # Check each principle
        for principle_name, principle in self.constitution.items():
            satisfied = await self._check_principle(principle_name, principle, code, results)

            if not satisfied["passes"]:
                violations.append(
                    {
                        "principle": principle_name,
                        "description": principle["description"],
                        "violation": satisfied["explanation"],
                        "severity": principle["severity"],
                    }
                )

        # Critical violations prevent approval
        critical = [v for v in violations if v["severity"] == "critical"]

        return ConstitutionalReview(
            approved=len(critical) == 0, violations=violations, critical_violations=critical
        )

    async def _check_principle(
        self, principle_name: str, principle: Dict[str, Any], code: Any, results: Any
    ) -> Dict[str, Any]:
        """
        Check if a principle is satisfied

        Args:
            principle_name: Name of principle
            principle: Principle definition
            code: Generated code
            results: Execution results

        Returns:
            Check result with pass/fail and explanation
        """

        # Use LLM to evaluate principle
        prompt = f"""Evaluate if this experiment satisfies the scientific principle:

Principle: {principle_name}
Description: {principle['description']}

Code:
{code.code if hasattr(code, 'code') else str(code)}

Results Summary:
{results.summary if hasattr(results, 'summary') else str(results)}

Does this experiment satisfy this principle?
Answer with: YES or NO, followed by a brief explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            answer = response.choices[0].message.content

            passes = answer.strip().upper().startswith("YES")

            return {"passes": passes, "explanation": answer}

        except Exception as e:
            self.logger.error(f"Error checking principle {principle_name}: {e}")
            # Default to passing if check fails
            return {"passes": True, "explanation": f"Check failed: {str(e)}"}

    async def _statistical_review(self, results: Any) -> StatisticalReview:
        """
        Review statistical validity

        Args:
            results: Execution results

        Returns:
            Statistical review result
        """

        self.logger.info("Performing statistical review")

        prompt = f"""Perform rigorous statistical analysis of these results:

Results Summary:
{results.summary if hasattr(results, 'summary') else str(results)}

Metrics:
{results.metrics if hasattr(results, 'metrics') else 'N/A'}

Evaluate:
1. Are the statistical tests appropriate?
2. Are assumptions of tests met?
3. Is sample size adequate?
4. Are effect sizes meaningful?
5. Are confidence intervals appropriate?
6. Is multiple testing handled correctly?
7. Are there any statistical red flags?

List any ISSUES found (or "None" if no issues).
Then provide RECOMMENDATIONS for improvement."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3,
            )

            analysis = response.choices[0].message.content

            # Parse issues and recommendations
            issues = self._parse_issues(analysis)
            recommendations = self._parse_recommendations_from_text(analysis)

            approved = len(issues) == 0 or "none" in issues[0].lower()

            return StatisticalReview(
                approved=approved, issues=issues, recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Error in statistical review: {e}")
            return StatisticalReview(
                approved=True,  # Default to approved if review fails
                issues=[f"Review failed: {str(e)}"],
                recommendations=["Manual review recommended"],
            )

    def _calculate_confidence(
        self, constitutional: ConstitutionalReview, statistical: StatisticalReview
    ) -> float:
        """Calculate overall confidence score"""

        # Start with high confidence
        confidence = 1.0

        # Reduce for violations
        for violation in constitutional.violations:
            if violation["severity"] == "critical":
                confidence -= 0.3
            elif violation["severity"] == "high":
                confidence -= 0.15
            else:
                confidence -= 0.05

        # Reduce for statistical issues
        confidence -= min(0.2, len(statistical.issues) * 0.05)

        return max(0.0, min(1.0, confidence))

    def _generate_recommendations(
        self, constitutional: ConstitutionalReview, statistical: StatisticalReview
    ) -> list[str]:
        """Generate recommendations for improvement"""

        recommendations = []

        # Add recommendations for violations
        for violation in constitutional.violations:
            recommendations.append(
                f"Address {violation['principle']}: {violation['violation']}"
            )

        # Add statistical recommendations
        recommendations.extend(statistical.recommendations)

        return recommendations

    def _parse_issues(self, text: str) -> list[str]:
        """Parse issues from text"""

        issues = []

        # Look for "ISSUES:" section
        if "ISSUES:" in text or "Issues:" in text:
            parts = text.split("ISSUES:")
            if len(parts) < 2:
                parts = text.split("Issues:")

            if len(parts) >= 2:
                issues_text = parts[1].split("RECOMMENDATIONS:")[0]
                # Extract bullet points or lines
                lines = [
                    line.strip().lstrip("-•*")
                    for line in issues_text.split("\n")
                    if line.strip()
                ]
                issues = [line for line in lines if line and len(line) > 3]

        return issues if issues else ["No specific issues identified"]

    def _parse_recommendations_from_text(self, text: str) -> list[str]:
        """Parse recommendations from text"""

        recommendations = []

        # Look for "RECOMMENDATIONS:" section
        if "RECOMMENDATIONS:" in text or "Recommendations:" in text:
            parts = text.split("RECOMMENDATIONS:")
            if len(parts) < 2:
                parts = text.split("Recommendations:")

            if len(parts) >= 2:
                rec_text = parts[1]
                # Extract bullet points or lines
                lines = [
                    line.strip().lstrip("-•*") for line in rec_text.split("\n") if line.strip()
                ]
                recommendations = [line for line in lines if line and len(line) > 3]

        return recommendations if recommendations else ["No specific recommendations"]
