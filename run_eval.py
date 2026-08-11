import json                        # to load our eval questions and save results
from agent import run_agent        # to reuse the actual agent we already built and tested
from groq import Groq
import os
import time
from llm_utils import safe_chat_completion

client = Groq(api_key=os.environ["GROQ_API_KEY"])

with open("eval_questions.json", "r", encoding="utf-8") as f:   # this opens our answer-key eval_questions.json file
    eval_set = json.load(f)                                      # to parse it into a Python list of dictionaries

def grade_answer(question, agent_answer, expected_answer, expected_citation):
    # this function is a separate AI call to grade or judge, it doesn't answer the question itself, it just compares the agent's real answer against the known-correct answer and decides whether to pass or fail
    # the judge is a separate AI call because we want to keep the agent's reasoning and the judge's grading completely independent, so the judge can't be influenced by the agent's reasoning or evidence-gathering process

    prompt = f"""Grade whether the AGENT ANSWER correctly answers the question and matches
the EXPECTED ANSWER in substance (doesn't need exact wording), and whether it cites
something matching the EXPECTED CITATION.

Respond with exactly one word first: PASS or FAIL. Then a one-sentence reason.

QUESTION: {question}
EXPECTED ANSWER: {expected_answer}
EXPECTED CITATION: {expected_citation}
AGENT ANSWER: {agent_answer}"""
    # this prompt hands the judge everything it needs: what was asked, what the correct answer looks like, what citation we expect, and what the agent actually said. then asks it to compare and grade

    response = safe_chat_completion(
        client,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content   # to return the judge's(Agents) decision as plain text

results = []   # will hold a detailed record for every question, to review later
passed = 0     # a running counter of how many questions passed

for item in eval_set:   # to loop through all 8 questions one at a time
    print(f"\nTesting: {item['question']}")

    agent_answer, _ = run_agent(item["question"])
    # run our real agent on this question, same one we've been testing all along.
    # run_agent returns two things (answer, evidence_log), but we only need the answer here, so the underscore "_" is Python's convention for receiving this value but deliberately ignoring it, since its not needed

    time.sleep(2)   # small pause between questions, Groq's limit is 30/minute so we don't need the long 15s pause Gemini needed

    grade = grade_answer(item["question"], agent_answer, item["expected_answer"], item["expected_citation"])
    # asks the judge function to grade this specific answer

    is_pass = grade.strip().upper().startswith("PASS")
    # .strip() removes any accidental leading/trailing whitespace
    # .upper() makes it uppercase so "pass" or "Pass" still counts correctly
    # .startswith("PASS") checks if the judge's response begins with that word
    # so all of this together is a reliable way to detect pass/fail even with tiny formatting differences

    if is_pass:
        passed += 1   # to increment the running score counter

    results.append({
        "question": item["question"],
        "agent_answer": agent_answer,
        "grade": grade,
    })
    # this saves full details for this question, so we can review exactly what happened later,
    # not just the final pass/fail count

    print(f"  Result: {grade.splitlines()[0]}")
    # .splitlines() breaks the judge's response into separate lines, and [0] grabs just the 1st line (which should be the PASS/FAIL word), 
    # so our final progress stays short instead of dumping the whole explanation

print(f"\n\n=== FINAL SCORE: {passed}/{len(eval_set)} ===")
# this is THE number that matters: how many questions passed out of the total, so you can see how well the agent did overall

with open("data/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
    # to save every question's full answer + grade to a file, so you can review exactly where it succeeded or failed, not just the final score



# eval_questions.json is our "answer key", there are 8 questions where we already know the correct answer and which PR should be cited, 
# based on what we found during testing. We use this to check if the agent gets things right, by comparing its real answers against these known-correct ones.