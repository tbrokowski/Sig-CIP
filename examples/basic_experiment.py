"""
Basic SciAgent Example

Demonstrates running a simple scientific experiment with SciAgent.
"""

import asyncio
from sciagent import SciAgent


async def main():
    """Run a basic experiment"""

    # Create SciAgent instance
    agent = SciAgent()

    print("🔬 Starting SciAgent experiment...\n")

    # Define research question
    query = "Test if dropout improves model generalization on CIFAR-10"

    # Run experiment in automated mode
    result = await agent.run(
        query=query,
        interactive=False,  # Automated mode - no approvals needed
    )

    # Display results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    print(f"\nExperiment ID: {result.experiment_id}")
    print(f"Query: {result.query}")
    print(f"Status: {result.state.status.value}")

    print("\n--- Hypothesis ---")
    print(result.planning.recommendation.hypothesis.statement)

    print("\n--- Analysis ---")
    print(result.analysis.summary[:500])

    print(f"\nSupports Hypothesis: {result.analysis.supports_hypothesis}")
    print(f"Confidence: {result.analysis.confidence:.2%}")

    print("\n--- Validation ---")
    print(f"Approved: {result.validation.approved}")
    print(f"Confidence: {result.validation.confidence:.2%}")

    if result.validation.recommendations:
        print("\n--- Recommendations ---")
        for i, rec in enumerate(result.validation.recommendations, 1):
            print(f"{i}. {rec}")

    if result.refinements:
        print("\n--- Proposed Refinements ---")
        for i, ref in enumerate(result.refinements, 1):
            print(f"{i}. {ref.description}")
            print(f"   Expected Info Gain: {ref.expected_information_gain:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
