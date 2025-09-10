from judgeval.tracer import Tracer

# Initialize the tracer with your project name
judgment = Tracer(project_name="default")

# Use the @judgment.observe decorator to trace the tool call
@judgment.observe(span_type="tool")
def my_tool():
    return "Hello world!"

# Use the @judgment.observe decorator to trace the function
@judgment.observe(span_type="function")
def sample_function():
    tool_called = my_tool()
    message = "Called my_tool() and got: " + tool_called
    return message

if __name__ == "__main__":
    res = sample_function()
    print(res)