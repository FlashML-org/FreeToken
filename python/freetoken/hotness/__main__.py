"""``python -m freetoken.hotness`` 入口：转发到 select 模块的 main。

Business Logic（为什么需要这个函数）:
    设计文档约定的调用形式是 ``python -m freetoken.hotness select ...``，
    包需要有 __main__ 才能以模块方式执行。

Code Logic（这个函数做什么）:
    导入选点模块的 main 并以其返回值作为进程退出码。
"""

import sys

if __name__ == "__main__":
    from freetoken.hotness.select import main

    sys.exit(main())
