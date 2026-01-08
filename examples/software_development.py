"""
Software Development with SciAgent

Demonstrates using SciAgent for software development tasks.
"""

import asyncio
from sciagent import SciAgent


async def main():
    """Use SciAgent for software development"""

    agent = SciAgent()

    print("💻 Using SciAgent for Software Development\n")

    # Software development task
    query = """
Build a FastAPI endpoint for user authentication with:
- JWT tokens
- Password hashing with bcrypt
- Rate limiting (max 5 requests per minute)
- Input validation with Pydantic
- PostgreSQL database integration

Generate complete, production-ready code with:
1. API endpoint implementation
2. Database models
3. Authentication logic
4. Unit tests with pytest
5. Error handling
"""

    print("Task:", query[:100] + "...\n")

    # Run in automated mode for faster development
    result = await agent.run(query=query, interactive=False)

    print("\n" + "=" * 60)
    print("DEVELOPMENT COMPLETE")
    print("=" * 60)

    print(f"\nExperiment ID: {result.experiment_id}")

    # Code generated
    print("\n--- Generated Code ---")
    print(f"Quality Score: {result.code.quality_score:.2%}")
    print(f"Iterations: {result.code.iterations}")

    # Show first 500 chars of code
    print("\nCode Preview:")
    print(result.code.code[:500])
    print("...")

    # Validation results
    print("\n--- Code Review ---")
    print(f"Approved: {result.validation.approved}")
    print(f"Confidence: {result.validation.confidence:.2%}")

    if result.validation.recommendations:
        print("\nRecommendations:")
        for rec in result.validation.recommendations[:3]:
            print(f"  • {rec}")

    # Save code to file
    output_file = f"generated_code_{result.experiment_id}.py"
    with open(output_file, "w") as f:
        f.write(result.code.code)

    print(f"\n✅ Code saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
