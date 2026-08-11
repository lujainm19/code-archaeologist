import os
from groq import Groq
from llm_utils import safe_chat_completion

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def verify_citations(answer, evidence_log):
    """Checks whether every claim in the answer is actually backed by the evidence gathered."""
    # this function is to be a skeptical second opinion, it doesn't answer the original question, it just checks the 1st answer's work

    evidence_text = "\n\n---\n\n".join(evidence_log)
    # join every piece of evidence together into one big text block, separated by dashed lines so it's easy to tell where one piece of evidence ends and the next begins

    prompt = f"""You are a strict fact-checker. Below is an AI agent's ANSWER, and the EVIDENCE
it actually gathered while researching. Check whether every factual claim in the answer is
genuinely supported by the evidence.

Respond with:
- "VERIFIED" if every claim is supported by the evidence
- Otherwise, list each unsupported or fabricated claim you find

EVIDENCE:
{evidence_text}

ANSWER TO CHECK:
{answer}"""
    # this prompt gives the model both the original answer and the raw proof, and asks it to compare them,
    #  it's not answering the question, it's grading someone else's(the agents) answer against the evidence

    response = safe_chat_completion(
        client,
        model="llama-3.3-70b-versatile",
        # a fast/cheap model is used here since fact-checking against provided text, is an easy task than the original research + reasoning was
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
    # returns the verification decision, either "VERIFIED" or a list of unsupported claims