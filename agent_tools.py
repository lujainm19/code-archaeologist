import os                                  # to read environment variables and build file paths
from github import Github, Auth            # PyGithub, to fetch PR data
from search_history import search_history  # reuse the search function we already built and tested

github_token = os.environ["GITHUB_TOKEN"]        # saved GitHub token
gh = Github(auth=Auth.Token(github_token))       # authenticate with GitHub
repo = gh.get_repo("encode/httpx")               # using an object representing this specific repo

def search_history_tool(query: str) -> str:
    """Search the httpx repository's commit and PR history for information relevant to a query.
    Use this to find WHY something was changed, discussions, or historical context.

    Args:
        query: A natural language question or topic to search for.
    """
    # this docstring isn't just for humans, Gemini reads it to understand when and why to use this tool, so it needs to be clear and descriptive

    results = search_history(query, n_results=5)   # reuse our already-built search function

    output = []   # will hold short, readable summaries of each result
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]          # the chunk's text
        meta = results["metadatas"][0][i]           # its type/tags
        output.append(f"[{meta.get('type')}] {text[:300]}")
        # only take the first 300 characters of each result, so we don't send
        # Gemini a huge wall of text for every single tool call

    return "\n\n".join(output)   # joining all results into one string to hand back to the agent

def read_file_tool(file_path: str) -> str:
    """Read the current contents of a specific file in the httpx source code.
    Use this to see the actual, current implementation of something.

    Args:
        file_path: Path relative to the repo, e.g. "httpx/_client.py"
    """
    full_path = os.path.join("data/httpx_repo", file_path)
    # build the full path by combining our local cloned repo's location with whatever relative path the AGENT decides to ask for

    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()   # read the WHOLE file as one string
        return content[:3000]    # cap it at 3000 characters so one huge file
                                  # doesn't blow up our context budget in one tool call
    except FileNotFoundError:
        # if the agent asks for a path that doesn't exist, just tell it clearly so it can try something else instead of crashing the whole agent
        return f"File not found: {file_path}"

def get_pr_tool(pr_number: int) -> str:
    """Fetch the full description and all comments of a specific pull request by number.
    Use this when you know a specific PR number and need its full discussion.

    Args:
        pr_number: The pull request number, e.g. 490
    """
    try:
        pr = repo.get_pull(pr_number)                        # fetch this specific PR live from GitHub
        comments = [c.body for c in pr.get_issue_comments()]  # grab every comment's text
        return f"PR #{pr.number}: {pr.title}\n{pr.body}\n\nComments:\n" + "\n---\n".join(comments[:5])
        # include only the first 5 comments, same reasoning as capping file length (to keep each individual tool result reasonably sized)
    except Exception as e:
        # catch any error (bad PR number, network issue, etc.) instead of crashing the whole agent over one failed tool call
        return f"Could not fetch PR #{pr_number}: {e}"
    
if __name__ == "__main__":
    print(search_history_tool("why was timeout handling added?")[:500])
    print("\n---\n")
    print(read_file_tool("httpx/_client.py")[:500])
    print("\n---\n")
    print(get_pr_tool(490)[:500])