# datasets.py
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.dataset import Dataset
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer

def build_initial_dataset() -> Dataset:
    examples = [
        Example(
            input="Question : What is the capital of the United States?",
            actual_output="Washington, D.C.",
            expected_output="Washington, D.C.",
            retrieval_context=["Washington, D.C. is the capital of the United States."],
            additional_metadata={"source": "seed"}
        ),
        Example(
            input="Question : How much wood could a wood chuck chuck?",
            actual_output="About as much as a woodchuck could chuck.",
            expected_output=None,
            retrieval_context=["Woodchucks don't actually chuck wood; it's a tongue twister."],
            additional_metadata={"source": "seed"}
        ),
    ]

    dataset = Dataset.create(
        name="interview_suite",
        project_name="bigone",
        examples=examples
    )
    # You can append later:
    dataset.add_examples([
        Example(
            input="Question : Who wrote The Republic?",
            actual_output="Plato.",
            expected_output="Plato.",
            retrieval_context=["The Republic is a philosophical dialogue by Plato."],
            additional_metadata={"source": "append"}
        )
    ])
    return dataset

def evaluate_dataset_sync(dataset: Dataset):
    client = JudgmentClient()
    results = client.run_evaluation(
        examples=dataset.examples,
        scorers=[
            AnswerRelevancyScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.6)
        ],
        model="gpt-4.1",
    )
    print(results)

if __name__ == "__main__":
    ds = build_initial_dataset()
    evaluate_dataset_sync(ds)
