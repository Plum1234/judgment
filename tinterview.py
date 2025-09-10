from judgeval.tracer import Tracer, wrap
from openai import OpenAI
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer

judgment = Tracer(project_name="Interview")
client = wrap(OpenAI())

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    return f"Question : {question}"

class SimpleAgent:
    def __init__(self, name: str):
        self.name = name

    @judgment.observe(span_type="function")
    @judgment.agent(identifier="name")
    def run_agent(self, prompt: str) -> str:
        task = format_question(prompt)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": task}]
        )
        answer = response.choices[0].message.content

        # ---- Online evals ----
        judgment.async_evaluate(
            scorer=AnswerRelevancyScorer(threshold=0.5),
            example=Example(input=task, 
                            actual_output=answer
                            ),
            model="gpt-4.1"
        )

        retrieval_ctx = []
        if retrieval_ctx:
            judgment.async_evaluate(
                scorer=FaithfulnessScorer(threshold=0.5),
                example=Example(
                    input=task, 
                    actual_output=answer, 
                    retrieval_context=retrieval_ctx
                ),
                model="gpt-4.1"
            )
        return answer

class Orchestrator:
    @judgment.observe(span_type="function")
    @judgment.agent()
    def run_agent(self):
        alice = SimpleAgent("Alice")
        bob = SimpleAgent("Bob")
        alice.run_agent("how much wood could a wood chuck chuck")
        bob.run_agent("how many logs per minute")

@judgment.observe(span_type="function")
def main():
    orchestrator = Orchestrator()
    orchestrator.run_agent()
main()