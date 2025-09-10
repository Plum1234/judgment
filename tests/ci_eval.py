# ci_eval.py
from judgeval import JudgmentClient
from judgeval.utils.file_utils import get_examples_from_yaml
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer

def run_yaml_suite(yaml_path: str = "tests.yaml"):
    client = JudgmentClient()
    examples = get_examples_from_yaml(yaml_path)
    results = client.run_evaluation(
        examples=examples,
        scorers=[
            AnswerRelevancyScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.6)
        ],
        model="gpt-4.1",
    )
    print(results)

def assert_yaml_suite(yaml_path: str = "tests.yaml"):
    client = JudgmentClient()
    client.assert_test(
        examples=get_examples_from_yaml(yaml_path),
        scorers=[
            AnswerRelevancyScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.6)
        ],
        model="gpt-4.1",
        project_name="bigone"
    )

if __name__ == "__main__":
    # Run without failing
    run_yaml_suite()

    # Uncomment for CI fail-fast gate:
    # assert_yaml_suite()
