"""一键冒烟入口：复用 tests/test_smoke.py 的用例，方便面试前手动跑。

运行：python smoke_test.py
等价于：pytest tests/test_smoke.py -q
"""

import sys

import pytest


def main() -> int:
    return pytest.main(["tests/test_smoke.py", "-q"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
