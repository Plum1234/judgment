from judgeval.tracer import Tracer
judgment = Tracer(project_name="default_project")

# send_message is defined outside of the agent class but is part of the call stack under the SimpleAgent.run() entry point method, so it will be associated with SimpleAgent.
class MessageClient:
    @judgment.observe(span_type="tool")
    def send_message(self, content: str) -> str:
        return f"Message sent with content: {content}"

# judgment.agent() will identify the agent based on the class name
# if no identifier is provided.
class OrchestratorAgent:
    @judgment.agent()
    @judgment.observe(span_type="function")
    def run(self):
        alice = SimpleAgent("Alice")  # agent will be identified as "Alice"
        bob = SimpleAgent("Bob")
        alice.run("Hello Bob, how are you?")
        bob.run("I'm good Alice, thanks for asking!")

# judgment.agent() specifies that the agents will be 
# identified based on their "name" attribute.
class SimpleAgent:
    def __init__(self, name: str):
        self.name = name
        self.message_client = MessageClient()

    @judgment.agent(identifier="name")
    @judgment.observe(span_type="tool")
    def run(self, content: str) -> str:
        return self.message_client.send_message(content)

@judgment.observe(span_type="function")
def main():
    orchestrator = OrchestratorAgent()
    orchestrator.run()
main()