import os                    # lets us read environment variables (brings in Python's built-in toolkit for talking to operating system)
from google import genai     # the Gemini toolkit we installed

api_key = os.environ["ARCHAEOLOGIST_API_KEY"]   # grab your saved key, (reaching into computer's stored environment variables (os.environ))
client = genai.Client(api_key=api_key)          # connect to Gemini using it

response = client.models.generate_content(      # send a prompt, get a reply
    model="gemini-2.5-flash",
    contents="Say hello in one short sentence."
)

print(response.text)   # show just the reply text