import time   # to pause between retries

def safe_chat_completion(client, **kwargs):
    """Wraps Groq's chat completion call with automatic retry on rate limits."""

    while True:
        try:
            return client.chat.completions.create(**kwargs)   # to make the actual API call

        except Exception as e:
            error_text = str(e)   # to convert the error to text so we can check what kind of error it is

            if "rate_limit" in error_text.lower() and "day" in error_text.lower():
                # to stop instead of looping forever since a daily limit won't resolve by itself
                print(f"  Hit a DAILY quota limit -- stopping instead of retrying forever.")
                raise

            # otherwise it's a short-term limit, so wait and try again
            print(f"  Hit an error ({e}), waiting 15s and retrying...")
            time.sleep(15)

# this file is created to wrap every Groq API call with automatic retry logic, so agent.py, verify_citations.py, and run_eval.py don't each need to repeat the same "try again if it fails" code.
# it waits and retries on short-term rate limits, but stops immediately for a daily quota error since that kind of limit won't fix itself no matter how many times we retry.