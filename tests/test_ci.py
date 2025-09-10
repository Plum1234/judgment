# test_ci.py
import pytest
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer
from agent import SimpleAgent

def test_relevancy_smoke():
    client = JudgmentClient()
    agent = SimpleAgent("CI-Agent")
    prompt = "What is the capital of the United States?"
    out = agent.run_agent(prompt)

    # Fail-fast if the model drifts
    client.assert_test(
        examples=[Example(input=f"Question : {prompt}", actual_output=out)],
        scorers=[AnswerRelevancyScorer(threshold=0.7)],
        model="gpt-4.1",
        project_name="bigone"
    )
