from openai import OpenAI
# from portkey_ai import Portkey

# portkey = Portkey(
#   base_url = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
#   api_key = "XXX+"
# )

# response = portkey.chat.completions.create(
#     model = "@vertexai/anthropic.claude-sonnet-4-6",
#     messages = [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "What is Portkey"}
#     ],
#     MAX_TOKENS = 512
# )

# print(response.choices[0].message.content)

# --- OpenAI：base_url + api_key 写死（自行替换占位符）---
openai_client = OpenAI(
    base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
    api_key="l7qxSRGWTdTZwGYw6F2M5L4dgYq+",
)

openai_response = openai_client.chat.completions.create(
    model="@vertexai/gemini-2.5-flash",
    # model="@vertexai/anthropic.claude-sonnet-4-6",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi in one short sentence."},
    ],
    max_tokens=128,
)

print(openai_response.choices[0].message.content)