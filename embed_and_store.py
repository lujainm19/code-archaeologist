import os                          # to read environment variables and file paths
import json                        # to load our chunks.json
import chromadb                    # the local vector database
from sentence_transformers import SentenceTransformer  # runs an embedding model locally on our own computer, so no API/internet needed after the first download

# to load the chunks we built in chunk_data.py, which contains all the commits, PRs, comments, and source code functions/classes
with open("data/chunks.json", "r", encoding="utf-8") as f:    # open chunks.json for reading, closes automatically when done
    chunks = json.load(f)                                     # parse the JSON text back into a Python list of dictionaries

print(f"Loaded {len(chunks)} chunks")   # len(chunks) is how many chunks we loaded, so it prints that count

# to load a small, fast embedding model. the first time this runs, it downloads the model (about 80MB) from the internet and saves it locally. 
# then every run after that just loads it from the disk, no internet needed
print("Loading local embedding model (first run downloads it, may take a minute)...")
model = SentenceTransformer("all-MiniLM-L6-v2")   # "all-MiniLM-L6-v2" is the specific model's name i choose to use

# to set up a local, persistent vector database (saves to disk in "data/chroma_db")
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="httpx_history")
# get_or_create_collection: makes a new "table" for the vectors, or reuses it if it already exists

# RESUME LOGIC: check how many chunks are already stored in the database,
# in case this script gets interrupted partway through and we need to restart later without redoing what we already finished
already_done = collection.count()
print(f"Already stored: {already_done} chunks -- resuming from there")

BATCH_SIZE = 100   # how many chunks we embed at once. bigger than before (was 20 - using api) because now we are embedding locally, so no API rate limit. can do bigger batches and fewer trips to the model, which is faster
                   # since there is now API rate limit (using local sentence-transformers) as it's all running locally on own computer now

# start the loop at "already_done" instead of 0, so if we're resuming, so we skip past chunks we've already embedded and stored
for batch_start in range(already_done, len(chunks), BATCH_SIZE):
    # range(start, stop, step) jumps by BATCH_SIZE each time: 0, 100, 200, 300...

    batch = chunks[batch_start : batch_start + BATCH_SIZE]   # slice out just this batch of chunks
    texts = [c["text"] for c in batch]                       # pull out just the "text" field from each chunk in this batch, as a list

    # this is the actual embedding step. model.encode() which takes the list of texts and
    # runs them through the local model, giving back one vector (list of numbers) per text.
    # it does this instantly, no waiting on a server, since it all happens on our own computer
    embeddings = model.encode(texts)

    # the model gives back its numbers in a special array format (numpy), but Chroma
    # needs plain Python lists, so .tolist() converts it into that normal list format
    embeddings = embeddings.tolist()

    ids = [str(batch_start + i) for i in range(len(batch))]
    # Chroma needs a unique string ID for every item we store, like a primary key.
    # this builds IDs like "0", "1", "2"... shifted by batch_start so every batch gets globally unique IDs instead of restarting at 0 each time

    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in batch]
    # for every chunk in this batch, this rebuilds its dictionary but WITHOUT the "text" key,
    # since Chroma stores the actual text separately in its own "documents" field below

    collection.add(          # this is the actual "save everything into the database" step
        ids=ids,              # the unique ID for each item
        embeddings=embeddings, # the actual vector (list of numbers) for each item
        documents=texts,       # the original readable text for each item
        metadatas=metadatas,   # the extra tags (type, author, sha, etc.) for each item
    )

    total_done = batch_start + len(batch)   # how many chunks we've completed in total so far
    print(f"Embedded and stored {total_done}/{len(chunks)} chunks")   # progress update, printed every batch since local embedding is fast and won't flood the terminal for long

print("Done! Vector store saved to data/chroma_db")   # confirms the whole process finished successfully




# The problem: i have 6987 chunks that all need to be turned into embeddings (a way of turning a piece of text into a list of numbers that represents its meaning), so that later i can search across all of them by MEANING instead of just exact keyword matches.
#
# i originally tried doing this through Gemini's embedding API, but hit two different rate limits: first a 100-requests-per-minute limit, then after fixing that with a pause between batches, a 1000-requests-per-day limit that would've taken about a week to fully get through.
#
# so instead, this script uses sentence-transformers, a package that runs a small embedding model, directly on my own computer. no api key, no internet needed after the one-time model download, and no rate limits at all since nothing leaves my machine.
#
# it processes chunks in batches of 100 (bigger than before since there's no limit to worry about), turns each batch's text into embeddings, and stores everything (the vector, the original text, and metadata like type/author/date) into a local Chroma vector database saved in data/chroma_db. that database is what the search step will query later to find relevant chunks for a question.