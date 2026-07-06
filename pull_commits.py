from pydriller import Repository    # to walk through a repo's commit history
import json                         # to us save data in JSON format
import os                           # to us create folders

REPO_URL = "https://github.com/encode/httpx"  # the repo i want to analyze for now (clone it into a temp folder)
OUTPUT_FILE = "data/commits.json"

commits_data = []   # this will hold one dictionary per commit

# traverse_commits() clones the repo (into a temp folder) and walks through every commit, oldest to newest
for commit in Repository(REPO_URL).traverse_commits():
    commits_data.append({
        "hash": commit.hash,
        "author": commit.author.name,
        "date": commit.author_date.isoformat(),
        "message": commit.msg,
        "files_changed": [f.filename for f in commit.modified_files],
    })

print(f"Pulled {len(commits_data)} commits") #len(commits_data) returns how many items are in the list so this prints Pulled # commits

os.makedirs("data", exist_ok=True)   # makes sure the "data" folder exists first

with open(OUTPUT_FILE, "w", encoding="utf-8") as f: # Opens the file for writing ("w" mode — creates it if missing, overwrites if it exists). encoding="utf-8" just ensures special characters are saved correctly, with ... as f: guarantees the file gets properly closed afterward
    json.dump(commits_data, f, indent=2) # Converts the whole Python list of dictionaries into JSON text and writes it directly into the open file f and indent=2

print(f"Saved to {OUTPUT_FILE}")