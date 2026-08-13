from openai import OpenAI

from core.config import config


class LLM:
    def __init__(self):
        if config.ALI_API_KEY:
            self.client = OpenAI(api_key=config.ALI_API_KEY, base_url=config.LLM_BASE_URL)
        else:
            self.client = None

    def generate(self, prompt):
        if not self.client:
            return "LLM未启用，请设置ALI_API_KEY"
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
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
