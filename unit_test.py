from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

client = JudgmentClient()

# agent = ...  # your agent
task = "What if these shoes don't fit?"
example = Example(
    input=task,
    actual_output=["All customers are eligible for a 44 day full refund at no extra cost."],  # e.g. "We offer a 30-day full refund at no extra cost."
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

scorer = FaithfulnessScorer(threshold=0.5)
results = client.run_evaluation(
    examples=[example],
    scorers=[scorer],
    model="gpt-4.1",
)
print(results)