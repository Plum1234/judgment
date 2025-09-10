import pytest
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

client = JudgmentClient()

def test_refund_policy_fails():
    task = "What if these shoes don't fit?"

    # Agent says 30 days, but context says 44 days
    example = Example(
        input=task,
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 44 day full refund at no extra cost."]
    )

    scorer = FaithfulnessScorer(threshold=0.5)

    # We EXPECT this to fail → so wrap it in pytest.raises
    with pytest.raises(AssertionError):
        client.assert_test(
            examples=[example],
            scorers=[scorer],
            model="gpt-4.1",
            project_name="default_project"
        )
