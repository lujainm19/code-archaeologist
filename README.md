# Code Archaeologist

An AI agent that answers questions about a codebase's history, with citations.

## How it works

1. `pull_commits.py` — pulls commit history from httpx(target repo) via pydriller
2. `pull_prs_issues.py` — pulls PR/issue history via GitHub API
3. `chunk_data.py` — splits commits, PRs, comments, and source code into approximately 7000 searchable chunks
4. `embed_and_store.py` — embeds chunks locally (sentence-transformers) into a Chroma vector database
5. `search_history.py` — semantic search over the vector store; given a question, returns the most relevant chunks

## Example queries tested
- "why was timeout handling added?" - found the actual PRs that introduced/refined timeout behavior
- "how does connection pooling work?" - found the founding PR that implemented it
- "SSL certificate verification" - found actual source code (`create_ssl_context`) purely by meaning