from judgeval.tracer import Tracer, wrap
from openai import OpenAI
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer

judgment = Tracer(project_name="bigone")
client = wrap(OpenAI())

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    return f"Question : {question}"

@judgment.observe(span_type="tool")
def retrieve_context(query: str):
    """
    Dummy retrieval; replace with your RAG/tool output.
    Returning a list enables FaithfulnessScorer.
    """
    if "capital" in query.lower():
        return ["Washington, D.C. is the capital of the United States."]
    if "wood" in query.lower():
        return ["Woodchucks (groundhogs) don't actually chuck wood; it's a tongue twister."]
    return []

class SimpleAgent:
    def __init__(self, name: str):
        self.name = name

    @judgment.agent(identifier="name")
    @judgment.observe(span_type="function")
    def run_agent(self, prompt: str) -> str:
        # Tools
        task = format_question(prompt)
        ctx = retrieve_context(prompt)

        # LLM call (auto-traced via wrap)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": task}],
            timeout=30
        )
        answer = (response.choices[0].message.content or "").strip()

        # ---- Online evals (attach to this trace) ----
        # Relevancy: is the answer responsive to the prompt?
        judgment.async_evaluate(
            scorer=AnswerRelevancyScorer(threshold=0.6),
            example=Example(
                input=task,
                actual_output=answer,
                # You can still tag evals with example-level metadata if you want:
                # additional_metadata={"agent": self.name, "user_prompt": prompt}
            ),
            model="gpt-4.1"
        )

        # Faithfulness: only if you have supporting context (RAG/tools)
        if ctx:
            judgment.async_evaluate(
                scorer=FaithfulnessScorer(threshold=0.6),
                example=Example(
                    input=task,
                    actual_output=answer,
                    retrieval_context=ctx
                ),
                model="gpt-4.1"
            )
        # ---------------------------------------------

        return answer

class Orchestrator:
    @judgment.agent()
    @judgment.observe(span_type="function")
    def run_agent(self):
        alice = SimpleAgent("Alice")
        bob = SimpleAgent("Bob")
        alice.run_agent("How much wood could a wood chuck chuck?")
        bob.run_agent("What is the capital of the United States?")

@judgment.observe(span_type="function")
def main():
    orchestrator = Orchestrator()
    orchestrator.run_agent()

if __name__ == "__main__":
    main()
