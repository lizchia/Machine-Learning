import openai
client = openai.OpenAI(
    api_key="sk-MPJDP7Wi0omLE0pvxFUJ0g",
    base_url="http://192.168.150.73:4000/v1" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="llama70b", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)