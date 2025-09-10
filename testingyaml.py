from judgeval import JudgmentClient
from judgeval.scorers import FaithfulnessScorer
from judgeval.utils.file_utils import get_examples_from_yaml

client = JudgmentClient()

def test_yaml_suite_fails():
    examples = get_examples_from_yaml("tests.yaml")
    scorer = FaithfulnessScorer(threshold=0.5)

    # This should FAIL because 30 vs 44 day mismatch
    client.assert_test(
        examples=examples,
        scorers=[scorer],
        model="gpt-4.1",
        project_name="default_project"
    )
