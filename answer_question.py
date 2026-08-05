import os                              # to read environment variables (our Gemini key)
from google import genai               # the Gemini toolkit we've been using
from search_history import search_history   # reuse the search function we already built and tested

api_key = os.environ["ARCHAEOLOGIST_API_KEY"]   # grab saved key
client = genai.Client(api_key=api_key)          # connect to Gemini using it

def answer_question(question):
    # this function takes a plain-english question, finds relevant chunks, and asks Gemini to answer using only those chunks, with citations

    results = search_history(question, n_results=8)
    # reusing our already-built search function, but asking for 8 results instead of 5, since we want a bit more context for the model to work here

    # build one big text block out of all the retrieved chunks, each labeled with a number so Gemini can reference "Source 3" etc. when it cites something
    context_parts = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]           # the chunk's actual text
        meta = results["metadatas"][0][i]            # its tags (type, author, sha, etc.)
        context_parts.append(f"[Source {i+1} | type: {meta.get('type')}]\n{text}")
        # each source gets a number + type label right above its text, so the model can point back to exactly which one it's citing

    context = "\n\n".join(context_parts)
    # to joinall the labeled sources into one string, with a blank line between each,
    # same join() idea we used earlier for files_changed

    # this is the actual instruction to the model - an f-string so we can drop our built context and the user's question directly into the prompt text
    prompt = f"""You are a code archaeologist. Answer the question using ONLY the sources below.
Cite sources by their number (e.g. "Source 3") for every claim. If the sources don't contain
enough information to answer, say so honestly instead of guessing.

SOURCES:
{context}

QUESTION: {question}"""

    response = client.models.generate_content(   # send the whole prompt (sources + question) to Gemini
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text   # retrieve just the model's written answer

# quick manual test, only runs when this file is executed directly 
if __name__ == "__main__":
    question = "Why was timeout handling added to httpx?"
    print(answer_question(question))   # run the whole pipeline end-to-end and print the answer