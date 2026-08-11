import json                        # to save/load data in JSON format
import os                          # for file paths and walking folders
import re                          # regular expressions, to detect function/class definitions in code

def load_json(path):               # small reusable helper since we load two files the same way
    with open(path, "r", encoding="utf-8") as f:   # open the file for reading, closes automatically when done
        return json.load(f)        # parse the JSON text back into a Python list/dict

def chunk_python_file(filepath, repo_root): # this function takes ONE .py file and splits it into one chunk per function/class

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f: # errors="ignore" skips any weird/broken characters instead of crashing on them
        lines = f.readlines()      # readlines() gives us a list, one string per line of the file

    pattern = re.compile(r'^(def |class )')
    # ^ means "start of line", so this matches lines that BEGIN with "def " or "class "
    # re.compile() pre-builds the pattern once so we can reuse it efficiently in the loop below

    boundaries = [i for i, line in enumerate(lines) if pattern.match(line)]
    # enumerate(lines) gives us (index, line) pairs as we loop
    # pattern.match(line) checks if THIS line starts a function/class
    # the list comprehension keeps only the INDEX (i) of lines where that's true
    # so "boundaries" ends up as a list of line numbers, e.g. [4, 20, 45], marking where each chunk starts

    file_chunks = []                # will hold every chunk found in this one file
    relative_path = os.path.relpath(filepath, repo_root)
    # converts a long full path into a short one relative to the repo root,
    # e.g. "C:/.../data/httpx_repo/httpx/_client.py" becomes just "httpx/_client.py"

    for idx, start in enumerate(boundaries):
        # idx = which boundary we're on (0, 1, 2...), start = the line number it starts at

        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        # ternary expression (value_if_true if condition else value_if_false):
        # if there's a NEXT boundary, this chunk ends right before the boundary starts
        # otherwise (this is the last function in the file), it ends at the file's last line

        chunk_lines = lines[start:end]     # slice out just the lines belonging to this one chunk
        code_text = "".join(chunk_lines)   # glue that list of line-strings back into one single string

        name_match = re.match(r'(def|class)\s+(\w+)', lines[start])
        # \s+ means "one or more spaces", \w+ means "one or more letters/numbers/underscore"
        # this pulls the actual function/class NAME out of its first line

        name = name_match.group(2) if name_match else "unknown"
        # .group(2) grabs the 2nd captured group (the name itself, not the def/class keyword)
        # guard against name_match being None (no match found) to avoid a crash, same idea as a null check

        file_chunks.append({
            "text": f"File: {relative_path}\n{code_text}",   # the actual chunk content, labeled with its file
            "type": "code",                                  # tags this chunk as source code (vs commit/PR later)
            "file_path": relative_path,                       # which file this came from
            "function_or_class_name": name,                   # which function/class this is
            "start_line": start + 1,                          # +1 because humans count lines starting at 1, not 0
            "end_line": end,
        })

    return file_chunks             # hand back every chunk we built from this one file

commits = load_json("data/commits.json")   # load our previously saved commit history
prs = load_json("data/prs.json")           # load our previously saved PR history

chunks = []   # master list that will hold every chunk, from every source, all mixed together

# --- turn each commit into one chunk ---
for c in commits:
    text = f"Commit by {c['author']} on {c['date']}: {c['message']}\nFiles changed: {', '.join(c['files_changed'])}"
    # ', '.join(list) turns ["a.py", "b.py"] into the single string "a.py, b.py"
    chunks.append({
        "text": text,
        "type": "commit",       # tags this chunk as a commit (so we can filter/search by type later)
        "sha": c["hash"],       # the commit's unique hash, useful for citations
        "author": c["author"],
        "date": c["date"],
    })

# --- turn each PR into a description chunk, plus one chunk per comment ---
for pr in prs:
    text = f"PR #{pr['number']}: {pr['title']}\n{pr['body'] or ''}"
    # pr['body'] or '' : if body is None (no description written), fall back to an empty string
    # instead of crashing when we try to insert None into an f-string
    chunks.append({
        "text": text,
        "type": "pr_description",
        "pr_number": pr["number"],
        "state": pr["state"],           # "open", "closed", etc.
        "date": pr["created_at"],
    })

    for i, comment in enumerate(pr["comments"]):
        # enumerate gives us both the index (i) and the comment text while looping
        chunks.append({
            "text": f"Comment on PR #{pr['number']} ({pr['title']}): {comment}",
            "type": "pr_comment",       # separate type from pr_description, so each comment is individually searchable
            "pr_number": pr["number"],  # links this comment back to its parent PR
            "comment_index": i,         # which comment number this was (0, 1, 2...) on that PR
        })

# --- walk the actual httpx source code and chunk every .py file by function/class ---
repo_root = "data/httpx_repo/httpx"   # this is the folder where httpx's actual package code lives

for root, dirs, files in os.walk(repo_root):
    # os.walk visits every folder recursively inside the cloned httpx repo, giving us (current_folder, subfolders, files) each time, and for every file ending in .py, it calls our chunking function on it and adds the results to the master chunks list
    for filename in files:
        if filename.endswith(".py"):   # only process actual Python source files, skip everything else
            filepath = os.path.join(root, filename)   # build the full path to this file
            chunks.extend(chunk_python_file(filepath, repo_root))
            # .extend() adds each item from the returned list individually into "chunks"
            # (different from .append(), which would nest the whole list as ONE item instead)

print(f"Built {len(chunks)} chunks total")   # len() = how many chunks we ended up with, across all sources

with open("data/chunks.json", "w", encoding="utf-8") as f:   # open (create) chunks.json for writing
    json.dump(chunks, f, indent=2)   # write everything as formatted JSON, indent=2 for readability

print("Saved to data/chunks.json")   # confirms the file was written successfully