from openai import OpenAI
from config import OPENAI_API_KEY, BASE_URL

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

def generate(prompt):

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def expand_query(query):
    prompt = f"""
请对下面的问题生成3个不同表达方式，每行一个，不要编号：

问题：{query}
"""
    result = generate(prompt)

    queries = [q.strip() for q in result.split("\n") if q.strip()]
    return queries[:3]