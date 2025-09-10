from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
from tracertest import run_agent

client = JudgmentClient()

task = "What is the capital of the United States?"
example = Example(
    input=task,
    actual_output=run_agent(task),  # e.g. "The capital of the U.S. is Washington, D.C. run_agent(task)," 
    retrieval_context=["Washington D.C. was founded in 1790 and became the capital of the U.S."],
)
 
scorer = FaithfulnessScorer(threshold=0.5)
client.run_evaluation(
    examples=[example],
    scorers=[scorer],
    model="gpt-4.1",
)