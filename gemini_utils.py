import time
from google import genai

def safe_generate_content(client, **kwargs):
    """Wraps generate_content with automatic retry on rate limits."""
    while True:
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            error_text = str(e)
            if "PerDay" in error_text:
                # a DAILY quota won't reset soon -- retrying is pointless, stop immediately
                print(f"  Hit a DAILY quota limit -- this won't resolve by waiting. Stopping.")
                raise   # re-raise the error so the script stops here instead of looping forever
            print(f"  Hit an error ({e}), waiting 60s and retrying...")
            time.sleep(60)

# gemini_utils.py is a shared helper that wraps every Gemini API call with automatic retry logic, so if we hit a rate limit, it waits 60 seconds and tries again instead of crashing the whole script.
# put it in its own file so agent.py, verify_citations.py, and run_eval.py can all reuse the same retry code instead of copy-pasting it three times.