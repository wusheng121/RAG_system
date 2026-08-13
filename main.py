"""命令行 RAG 问答 demo（复用 rag_service.ask，避免逻辑分叉）。

运行：python main.py
"""

from services.rag_service import ask


def main() -> None:
    print("=== LLM + RAG 问答系统 ===")
    print("输入问题，按回车查看答案，输入 exit 退出")
    while True:
        query = input("\n请输入问题：").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        print("\n【回答】")
        print(ask(query))


if __name__ == "__main__":
    main()
