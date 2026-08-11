import os
import json
from groq import Groq
from agent_tools import search_history_tool, read_file_tool, get_pr_tool, TOOL_SCHEMAS   # the three tools + their schemas
from llm_utils import safe_chat_completion

client = Groq(api_key=os.environ["GROQ_API_KEY"])

tools = [search_history_tool, read_file_tool, get_pr_tool]
# these are the real Python functions we call; TOOL_SCHEMAS (imported above) is the separate
# JSON description of them that we hand to Groq, since Groq (unlike Gemini) can't auto-build
# a schema from docstrings, we wrote that schema by hand in agent_tools.py

tool_map = {t.__name__: t for t in tools}
# a dictionary comprehension: builds {"search_history_tool": <function>, ...}
# so later, given just a name (a string the model gives us), we can look up and actually call the real function like a Java Map<String, Function>
# tool_map is a lookup table (like a Java Map<String, Function>) that allows us to get the actual function back when you only have its name as a string, which is exactly what the model gives you when it wants to call a tool. The line {t.__name__: t for t in tools} builds that table automatically: for every function t in your list, it uses the function's own name (t.__name__) as the key and the function itself as the value.

SYSTEM_INSTRUCTION = """You are a code archaeologist investigating the httpx repository.
Use the available tools to research the answer before responding.
Always cite your sources (PR numbers, file names) in your final answer.
If you can't find enough information, say so honestly."""
# this sets the agent's overall behavior/personality for the whole conversation, aside from the actual question being asked



def run_agent(question, max_turns=6):
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": question},
    ]
    # this list is the conversation history (we keep appending to it as we go) since the model
    # needs the full conversation each time, it has no memory of its own. Groq's format is justn plain dictionaries

    evidence_log = []
    # this keeps a running list of everything the agent actually looked up during its research,
    # every tool it called and what that tool returned. we need this so we can later check
    # "did the agent's final answer actually match what it really found, or did it make something up?"

    for turn in range(max_turns):
        response = safe_chat_completion(
            client,
            model="llama-3.3-70b-versatile",
            messages=messages,                # the whole conversation so far
            tools=TOOL_SCHEMAS,                # tell the model what tools are available
        )

        message = response.choices[0].message   # the model's response for this turn
        messages.append(message)                 # add its response to our running history

        if not message.tool_calls:
            # when the model makes no tool calls this turn, meaning it's done researching
            # and is giving its final, plain-text answer, so this returns both the answer text and the full evidence trail we collected
            return message.content, evidence_log

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            # Groq sends arguments back as a JSON-formatted string, not a ready to use dict, so we have to parse it ourself

            print(f"  [Agent is calling: {tc.function.name}({args})]")

            result = tool_map[tc.function.name](**args)   # to actually run the real tool function

            evidence_log.append(f"Called {tc.function.name}({args}):\n{result}")
            # save a record of exactly what was called, with what arguments, and what it returned, this becomes the "proof" for fact-checking later

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
            # to add the tool's result to the conversation, tagged with which tool call it answers, then the loop goes back to the top and asks the model again, so now it has real information to work with

    return "Reached max turns without a final answer.", evidence_log
    # this is so that even the "gave up" case still returns whatever evidence was gathered so far,
    # safety net in case the agent never settles on a final answer within max_turns

if __name__ == "__main__":
    from verify_citations import verify_citations   # fact-checking function

    question = "Why was connection pooling implemented in httpx?"
    print(f"QUESTION: {question}\n")

    answer, evidence_log = run_agent(question)
    # run_agent returns two values, so we unpack both here (now gets 2 things back, not just the answer)
    # this is how in python you receive multiple return values at once, similar to returning a small custom object/tuple in Java and reading two fields off it

    print(f"\nFINAL ANSWER:\n{answer}")

    print(f"\n--- VERIFYING CITATIONS ---")
    verification = verify_citations(answer, evidence_log)   # to run fact-checker
    print(verification)





# We give the model the question, our system instructions, and the list of 3 tools it's allowed to use.
# The model looks at the question and decides either answer directly, or say "I want to call this tool with these arguments."
# If it wants a tool, we actually run that real Python function, take its result, and hand that result back to the model as new information.
# We repeat this in a loop, so the model can keep asking for more tools, one after another, until it finally responds with a plain text answer instead of a tool request, and we return that as the final answer.
# After the agent gives its final answer, then we run a separate fact-checking step that compares the answer against everything the agent actually looked up, to catch any claims that aren't really backed by evidence.