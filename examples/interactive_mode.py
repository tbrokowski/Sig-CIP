"""
Interactive SciAgent Example

Demonstrates using SciAgent with custom approval callbacks.
"""

import asyncio
from sciagent import SciAgent
from sciagent.utils.models import HumanApprovalRequest


async def custom_approval_handler(request: HumanApprovalRequest):
    """
    Custom approval handler that decides based on context

    This demonstrates how you can implement custom logic for approvals,
    such as auto-approving certain types of actions or sending notifications.
    """

    print(f"\n{'=' * 60}")
    print(f"APPROVAL REQUEST: {request.request_type}")
    print(f"{'=' * 60}")

    if request.request_type == "approval":
        # Simple yes/no approval
        message = request.context.get("message", "Approve?")
        print(f"\n{message}")

        # For this example, auto-approve low-risk actions
        if "data" in message.lower():
            print("→ Auto-approved (data preparation)")
            response = "yes"
        else:
            # Ask user
            user_input = input("\nApprove? (yes/no): ")
            response = user_input.lower()

        request.response.set_result(response)

    elif request.request_type == "choice":
        # Multiple choice
        message = request.context["message"]
        options = request.options

        print(f"\n{message}\n")

        for i, option in enumerate(options):
            print(f"  [{i+1}] {option}")

        choice = input("\nSelect option (number): ")
        selected = options[int(choice) - 1]

        request.response.set_result(selected)


async def main():
    """Run experiment with interactive approvals"""

    # Create SciAgent instance
    agent = SciAgent()

    print("🔬 Starting SciAgent in INTERACTIVE mode...\n")

    # Define research question
    query = "Compare different optimizers (SGD vs Adam) on MNIST"

    # Run experiment with custom approval handler
    result = await agent.run(
        query=query,
        interactive=True,
        on_approval=custom_approval_handler,
    )

    # Display results
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)

    print(f"\nExperiment ID: {result.experiment_id}")
    print(f"Status: {result.state.status.value}")

    print("\n--- Analysis Summary ---")
    print(result.analysis.summary[:300] + "...")

    print(f"\nConfidence: {result.analysis.confidence:.2%}")
    print(f"Approved: {result.validation.approved}")


if __name__ == "__main__":
    asyncio.run(main())
