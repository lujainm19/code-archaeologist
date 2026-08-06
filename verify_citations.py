import os
from google import genai

api_key = os.environ["ARCHAEOLOGIST_API_KEY"]
client = genai.Client(api_key=api_key)

def verify_citations(answer, evidence_log):
    """Checks whether every claim in the answer is actually backed by the evidence gathered."""
    # this function's ONLY job is to be a skeptical second opinion -- it doesn't
    # answer the original question at all, it just checks the FIRST answer's work

    evidence_text = "\n\n---\n\n".join(evidence_log)
    # glue every piece of evidence together into one big text block, separated by
    # dashed lines so it's easy to tell where one piece of evidence ends and the next begins

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
    # this prompt gives the model BOTH the original answer AND the raw proof,
    # and asks it to compare them critically -- it's not answering the question,
    # it's grading someone else's answer against the evidence

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        # a fast/cheap model is fine here since fact-checking against provided text
        # is an easier task than the original research + reasoning was
        contents=prompt,
    )

    return response.text   # hand back the verification verdict
