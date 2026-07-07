import os                          # to read environment variables (our GitHub token)
import json                        # to save data in JSON format
from github import Github, Auth    # PyGithub's main class, to talk to GitHub's API

token = os.environ["GITHUB_TOKEN"]   # to grab saved GitHub token
gh = Github(auth=Auth.Token(token))  # authenticate with GitHub using that token
repo = gh.get_repo("encode/httpx")   # to grab an object representing this specific repo

# Check how many API requests we have left before we even start —
# this tells us if rate limiting is even a risk right now
rate = gh.get_rate_limit()
print(f"Rate limit remaining: {rate.resources.core.remaining}/{rate.resources.core.limit}")

prs_data = []      # will hold one dictionary per pull request
issues_data = []   # will hold one dictionary per issue (non-PR (pull request - when someone is proposing an actual code change ) item)

# get_issues(state="all") fetches both open AND closed issues/PRs — by default
# GitHub only gives you open ones, so we have to explicitly ask for "all"
count = 0
for item in repo.get_issues(state="all"):
    entry = {
        "number": item.number,               # the #123 number seen on GitHub
        "title": item.title,                 # the title text
        "body": item.body,                   # the main description text someone wrote
        "state": item.state,                 # "open" or "closed"
        "created_at": item.created_at.isoformat(),   # convert datetime -> string, same reason as commits.py
        "comments": [c.body for c in item.get_comments()],  # to grab juSst the text of every comment
    }

    # GitHub quirk: every PR/pull request is secretly ALSO an "issue" internally.
    # item.pull_request only exists (is not None) if this "issue" is actually a PR.
    # None so this checks "does this have PR data attached?"
    if item.pull_request is not None:
        prs_data.append(entry)     # it's a PR ((has code attached))
    else:
        issues_data.append(entry)  # it's a genuine issue (plain)

    count += 1
    if count % 200 == 0: # print progress every 200 items instead of staying silent to confirm the script is still running (GitHub repos can have thousands of issues/PRs)
        print(f"Processed {count} items so far...")

print(f"Pulled {len(prs_data)} PRs and {len(issues_data)} issues")  # f-string, len(prs_data) returns how many items are in the list so this # prints how many PRs and issues we collected

os.makedirs("data", exist_ok=True)   # make sure the data/ folder exists (probably already does from commits.py), create the data folder if it doesn't already exist

with open("data/prs.json", "w", encoding="utf-8") as f:   # open prs.json for writing, and closes automatically when done
    json.dump(prs_data, f, indent=2)                      # writes the PR list into it as formatted JSON with indent=2 for readability

with open("data/issues.json", "w", encoding="utf-8") as f:   # open (create) issues.json for writing
    json.dump(issues_data, f, indent=2)                      # writes the issues list into it as formatted JSON

print("Saved to data/prs.json and data/issues.json")  # confirms both files were written

# Issues tell about the problem that triggered a change 
# PRs tell about the solution and the reasoning behind it so need both to understand the full context of a change.