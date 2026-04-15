from openai import OpenAI
import os

class LLM:
    def __init__(self):
        api_key = os.getenv("ALI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            self.client = None

    def generate(self, prompt):
        if not self.client:
            return "LLM未启用，请设置ALI_API_KEY"
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM调用失败: {e}"

    def expand_query(self, query):
        prompt = f"请对下面的问题生成3个不同表达方式，每行一个，不要编号：\n\n问题：{query}"
        result = self.generate(prompt)
        if not result or result.startswith("LLM"):
            return [query]
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        return queries[:3] if queries else [query]
