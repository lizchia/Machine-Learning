import openai
import os
from dotenv import load_dotenv

if not load_dotenv():
    print("no .env")

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
base_url = os.getenv("BASE_URL")

client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model=model_name, # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)