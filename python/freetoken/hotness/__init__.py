"""热点专家选点工具包：纯 Python 的 stats 读取、top-K 选点与 pin list 输出。

与运行期计数器 ``freetoken.moe.hotness`` 互不依赖；本包顶层不引入 torch
（torch 仅在 select 的 --budget-gib 分支内延迟导入）。对 ``main`` /
``SCHEMA_VERSION`` 采用 PEP 562 惰性导出，保证 ``python -m
freetoken.hotness.select`` 直接执行时不会因包初始化抢先导入本模块而触发
runpy 的 RuntimeWarning。
"""

__all__ = ["SCHEMA_VERSION", "main"]


def __getattr__(name: str):
    """
    Business Logic（为什么需要这个函数）:
        包的公共入口需要同时支持 ``from freetoken.hotness import main`` 与
        ``python -m freetoken.hotness.select`` 两种用法；惰性导出让后者在
        runpy 以 __main__ 执行 select 前不产生第二次 select 导入。

    Code Logic（这个函数做什么）:
        仅当属性名在 __all__ 中时才导入 freetoken.hotness.select 并返回对应
        属性，其余名称按惯例抛 AttributeError。
    """
    if name in __all__:
        from freetoken.hotness import select

        return getattr(select, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
