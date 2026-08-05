import chromadb                                          # the local vector database
from sentence_transformers import SentenceTransformer    # to load the same local embedding model we used to build the database

# load the SAME model we used to embed all our chunks earlier 
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to the persistent database built (chroma_db) and saved to disk
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="httpx_history")
# get_or_create_collection: reconnects to the collection from embed_and_store.py instead of creating a new empty one

def search_history(query, n_results=5):
    # this function takes a plain-English question (query) and returns the n_results most relevant chunks from our vector database

    query_embedding = model.encode([query]).tolist()
    # turn the question itself into an embedding, using the exact same model we used for all the stored chunks
    # wrapped in [query] because encode() takes a list of texts

    results = collection.query(
        query_embeddings=query_embedding,   # search using this embedding
        n_results=n_results,                # how many top matches to return
    )
    # this is the search step where Chroma compares our query's embedding to every stored embedding and returns the closest matches

    return results   # hand back the raw results so other code (or our test below) can use them

# quick manual test, only runs when this file is executed directly with "python search_history.py"
# if another file later does "from search_history import search_history" to reuse the function,
# this block below will not run -- it's only for testing this file on its own

if __name__ == "__main__":

    # to tests whether search holds up across a variety of questions, we can try a few different queries and see what results come back. the goal is to see if it finds relevant chunks even when the wording is different from the original text, or when the question is more about "why" something happened rather than "what" happened.
    test_queries = [
        "why was timeout handling added?",           # historical "why" question (already tested, worked well)
        "how does the connection pooling work?",      # should retrieve the actual code, not just history
        "what bugs were reported about redirects?",    # tests whether it finds relevant discussion, even with different wording
        "SSL certificate verification",                 # a more keyword-like query (not even a full sentence), testing a different phrasing style
    ]

    for query in test_queries:   # loop through each test question one at a time

        print(f"\n{'='*60}")     # '='*60 repeats the "=" character 60 times, just a visual separator to make each query's results easy to tell apart in the terminal output
        print(f"QUERY: {query}")
        print('='*60)

        results = search_history(query)   # to run search function on all the questions

        # results["documents"][0] is a list of the matched text, the [0] is because Chroma supports searching multiple queries at once, and we only sent one, so everything relevant lives at index 0
        for i in range(len(results["documents"][0])):
            text = results["documents"][0][i]         # the actual chunk text for this result
            metadata = results["metadatas"][0][i]     # its tags: type, author, sha, etc.
            distance = results["distances"][0][i]     # how "far" this result is from our question (smaller = more relevant)

            print(f"\n--- Result {i+1} (distance: {distance:.3f}) ---")   # {distance:.3f} formats the number to 3 decimal places inside the f-string
            print(f"Type: {metadata.get('type')}")   # what kind of chunk this is (commit, pr_description, pr_comment, or code)
            print(text[:200])   # print only the first 200 characters, so we don't flood the terminal with one giant chunk


