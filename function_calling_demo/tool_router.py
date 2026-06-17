"""
Tool Router 完整实现 —— 分类 → 选择 → 重试 → 降级
"""

import time
from typing import Any, Callable, Optional


class Tool:
    """工具定义"""

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        func: Callable,
        retry: int = 2,
        timeout: float = 10.0,
        fallback: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.category = category
        self.func = func
        self.retry = retry
        self.timeout = timeout
        self.fallback = fallback


class ToolRouter:
    """工具路由：分类 → 选择 → 执行 → 降级"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}
        self._intent_keywords: dict[str, list[str]] = {
            "weather": ["天气", "温度", "下雨", "气温", "湿度"],
            "search": ["搜索", "查找", "查询", "搜一下", "找"],
            "calendar": ["日程", "会议", "提醒", "安排", "日历"],
            "database": ["数据库", "查数据", "记录", "存储"],
            "general": [],
        }

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._categories.setdefault(tool.category, []).append(tool.name)

    # ========== 第1层：意图分类 ==========

    def classify_intent(self, query: str) -> str:
        for category, keywords in self._intent_keywords.items():
            if any(kw in query for kw in keywords):
                print(f"    [路由] 意图匹配: {category}")
                return category
        print(f"    [路由] 意图匹配: general（兜底）")
        return "general"

    # ========== 第2层：工具选择 ==========

    def select_tools(self, query: str, max_tools: int = 5) -> list[Tool]:
        category = self.classify_intent(query)
        tool_names = self._categories.get(category, [])
        general_names = self._categories.get("general", [])
        all_names = tool_names + [n for n in general_names if n not in tool_names]
        selected = [self._tools[n] for n in all_names[:max_tools]]
        print(f"    [路由] 选中工具: {[t.name for t in selected]}")
        return selected

    def format_tools_prompt(self, tools: list[Tool]) -> str:
        lines = ["可用工具："]
        for t in tools:
            lines.append(f"  - {t.name}: {t.description} [{t.category}]")
        return "\n".join(lines)

    # ========== 第3层：执行（重试 + 降级） ==========

    def execute(self, name: str, **kwargs) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"未知工具: {name}")

        last_error = None

        # 重试阶段
        for attempt in range(1, tool.retry + 1):
            try:
                result = tool.func(**kwargs)
                print(f"    [执行] {tool.name} 第{attempt}次 → 成功")
                return {"data": result, "source": "primary"}
            except Exception as e:
                last_error = e
                print(f"    [执行] {tool.name} 第{attempt}次 → 失败: {e}")
                if attempt < tool.retry:
                    wait = 2 ** attempt
                    print(f"    [重试] 等待 {wait}s 后重试...")
                    time.sleep(wait)

        # 降级阶段
        if tool.fallback:
            fallback_tool = self._tools.get(tool.fallback)
            if fallback_tool:
                try:
                    result = fallback_tool.func(**kwargs)
                    print(f"    [降级] 切换到备用工具 {tool.fallback} → 成功")
                    return {"data": result, "source": "fallback"}
                except Exception as e:
                    last_error = e
                    print(f"    [降级] 备用工具也失败: {e}")

        raise RuntimeError(f"工具 [{name}] 调用失败: {last_error}")


# ============================================================
# 模拟工具函数
# ============================================================

def mock_weather(city: str) -> str:
    """正常天气查询"""
    return f"{city} 25°C 晴，东南风3级"

fail_count = {"get_weather": 0}

def mock_unstable_weather(city: str) -> str:
    """不稳定天气：前2次失败，第3次成功"""
    fail_count["get_weather"] = fail_count.get("get_weather", 0) + 1
    n = fail_count["get_weather"]
    if n <= 2:
        raise ConnectionError(f"API超时（第{n}次）")
    return f"{city} 22°C 多云"

def mock_weather_fallback(city: str) -> str:
    """备用天气查询"""
    return f"{city} 20°C（备用数据源）"

fail_count_always = {"search": 0}

def mock_always_fail(q: str) -> str:
    """永远失败的搜索"""
    fail_count_always["search"] = fail_count_always.get("search", 0) + 1
    n = fail_count_always["search"]
    raise TimeoutError(f"搜索服务超时（第{n}次）")


# ============================================================
# 测试
# ============================================================

def test_basic_route():
    """测试1：基本路由 + 分类"""
    print("\n" + "=" * 50)
    print("测试1：基本路由")
    print("=" * 50)

    router = ToolRouter()
    router.register(Tool(
        name="get_weather", description="查询城市天气", category="weather",
        func=mock_weather,
    ))
    router.register(Tool(
        name="search_web", description="搜索互联网信息", category="search",
        func=lambda q: f"关于「{q}」的搜索结果：...",
    ))

    query = "上海明天天气"
    tools = router.select_tools(query)
    print(router.format_tools_prompt(tools))

    result = router.execute("get_weather", city="上海")
    print(f"  最终结果: {result}")
    assert result["data"] == "上海 25°C 晴，东南风3级"
    print("  [OK] 测试1通过\n")


def test_retry():
    """测试2：重试机制（前2次失败，第3次成功）"""
    print("=" * 50)
    print("测试2：重试指数退避")
    print("=" * 50)

    router = ToolRouter()
    fail_count["get_weather"] = 0
    router.register(Tool(
        name="get_weather", description="查询城市天气", category="weather",
        func=mock_unstable_weather, retry=3,
    ))

    result = router.execute("get_weather", city="北京")
    print(f"  最终结果: {result}")
    assert result["data"] == "北京 22°C 多云"
    print("  [OK] 测试2通过\n")


def test_fallback():
    """测试3：重试全部失败 → 降级到备用工具"""
    print("=" * 50)
    print("测试3：降级（fallback）")
    print("=" * 50)

    router = ToolRouter()
    fail_count["get_weather"] = 0
    router.register(Tool(
        name="get_weather", description="查询城市天气", category="weather",
        func=mock_unstable_weather, retry=3, fallback="get_weather_backup",
    ))
    router.register(Tool(
        name="get_weather_backup", description="备用天气查询", category="weather",
        func=mock_weather_fallback,
    ))
    # 改 retry=1，第1次失败后直接触发降级
    fail_count["get_weather"] = 0
    router._tools["get_weather"].retry = 1

    result = router.execute("get_weather", city="广州")
    print(f"  最终结果: {result}")
    assert result["source"] == "fallback"
    print("  [OK] 测试3通过\n")


def test_all_fail():
    """测试4：所有方案都失败 → 报错"""
    print("=" * 50)
    print("测试4：所有方案都失败")
    print("=" * 50)

    router = ToolRouter()
    fail_count_always["search"] = 0
    router.register(Tool(
        name="search_web", description="搜索", category="search",
        func=mock_always_fail, retry=2,
    ))

    try:
        router.execute("search_web", q="AI")
        print("  [FAIL] 应该抛出异常但没抛出")
    except RuntimeError as e:
        print(f"  正确捕获异常: {e}")
        print("  [OK] 测试4通过\n")


def test_100_tools():
    """测试5：模拟100个工具的场景"""
    print("=" * 50)
    print("测试5：模拟100个工具")
    print("=" * 50)

    router = ToolRouter()

    # 注册大量天气工具
    for i in range(30):
        router.register(Tool(
            name=f"weather_detail_{i}",
            description=f"天气详情{i}",
            category="weather",
            func=lambda: f"结果{i}",
        ))

    # 注册大量搜索工具
    for i in range(30):
        router.register(Tool(
            name=f"search_detail_{i}",
            description=f"搜索详情{i}",
            category="search",
            func=lambda: f"结果{i}",
        ))

    # 注册大量数据库工具
    for i in range(40):
        router.register(Tool(
            name=f"db_query_{i}",
            description=f"数据库查询{i}",
            category="database",
            func=lambda: f"结果{i}",
        ))

    print(f"  总共注册: {len(router._tools)} 个工具")

    # 用户问天气 → 路由只选中 weather 类的前5个
    tools = router.select_tools("今天广州会下雨吗", max_tools=5)
    print(f"  路由后：只给 LLM 看 5 个工具（从30个weather中选出）")
    assert all(t.category == "weather" for t in tools)
    print("  [OK] 测试5通过\n")


if __name__ == "__main__":
    test_basic_route()
    test_retry()
    test_fallback()
    test_all_fail()
    test_100_tools()
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
