import os
from google import genai
from google.genai import types
from agent_tools import search_history_tool, read_file_tool, get_pr_tool   # the three tools

api_key = os.environ["ARCHAEOLOGIST_API_KEY"]
client = genai.Client(api_key=api_key)

tools = [search_history_tool, read_file_tool, get_pr_tool]
# passing the actual Python functions directly (the SDK builds Gemini's tool chema from their docstrings/type hints automatically, no manual written JSON needed)

tool_map = {t.__name__: t for t in tools}
# a dictionary comprehension: builds {"search_history_tool": <function>, ...}
# so later, given just a name (a string Gemini gives us), we can look up and actually call the real function  like a Java Map<String, Function>
# tool_map is a lookup table (like a Java Map<String, Function>) that allows us to get the actual function back when you only have its name as a string, which is exactly what Gemini gives you when it wants to call a tool. The line {t.__name__: t for t in tools} builds that table automatically: for every function t in your list, it uses the function's own name (t.__name__) as the key and the function itself as the value.

SYSTEM_INSTRUCTION = """You are a code archaeologist investigating the httpx repository.
Use the available tools to research the answer before responding.
Always cite your sources (PR numbers, file names) in your final answer.
If you can't find enough information, say so honestly."""
# this sets the agent's overall behavior/personality for the whole conversation, aside from the actual question being asked

def run_agent(question, max_turns=6):
    # max_turns to limit the amt of back-and-forth steps the agent can take, so a confused agent can't loop forever burning API calls

    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    # this list is the conversation history(we keep appending to it as we go) since Gemini needs the full conversation each time, it has no memory of its own

    for turn in range(max_turns):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,                    # the whole conversation so far
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=tools,                       # tell Gemini what tools are available
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                # we're disabling the SDK's built-in auto-calling because we want to see and control every step of the loop ourselves, so we can log it and handle errors instead of Gemini just crashing if a tool fails
            ),
        )

        candidate = response.candidates[0]     # Gemini's response for this turn
        contents.append(candidate.content)     # add its response to our running history

        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
        # a response can contain multiple "parts", so this pulls out just the ones where Gemini decided to call a tool (function_call is None otherwise)

        if not function_calls:
            # when gemini makes no tool calls this turn, meaning it's done researching and is giving its final, plain-text answer
            return response.text

        # otherwise, gemini wants to use one or more tools, this let's it actually run them
        function_response_parts = []
        for fc in function_calls:
            print(f"  [Agent is calling: {fc.name}({dict(fc.args)})]")
            # a visible log so we can watch the agent's reasoning steps as they happen

            result = tool_map[fc.name](**fc.args)
            # to look up the real function by name, and then call it. **fc.args looks at gemini's provided arguments as keyword arguments, e.g. query="why..."

            function_response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result})
            )
            # to package the tool's result in the format Gemini expects back

        contents.append(types.Content(role="user", parts=function_response_parts))
        # to add the tool results to the conversation, then the loop goes back to the top and asks Gemini again. so now it has real information to work with

    return "Reached max turns without a final answer."
    # safety net in case the agent never settles on a final answer within max_turns

if __name__ == "__main__":
    question = "Why was connection pooling implemented in httpx?"
    print(f"QUESTION: {question}\n")
    answer = run_agent(question)
    print(f"\nFINAL ANSWER:\n{answer}")

# We give Gemini the question, our system instructions, and the list of 3 tools it's allowed to use.
# Gemini looks at the question and decides: either answer directly, or say "I want to call this tool with these arguments."
# If it wants a tool, we actually run that real Python function, take its result, and hand that result back to Gemini as new information.
# We repeat steps 2-3 in a loop, so gemini can keep asking for more tools, one after another, until it finally responds with a plain text answer instead of a tool request, and we return that as the final answer.