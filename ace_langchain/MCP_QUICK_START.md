# MCP 快速开始指南

## 📦 安装

```bash
pip install langchain-mcp-adapters
```

## 🚀 快速使用

### 最简单的方式

```python
from langchain_openai import ChatOpenAI
from ace_langchain import (
    LangChainGenerator,
    LangChainPlaybook,
    get_mcp_tools,  # 官方 MultiServerMCPClient 封装
)

# 1. 配置 MCP 服务器并获取工具
tools = await get_mcp_tools({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"],
    },
    "weather": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",
    }
})

# 2. 创建 Generator（带 MCP 工具）
model = ChatOpenAI(model="gpt-4o-mini")
generator = LangChainGenerator(model=model, tools=tools)

# 3. 使用 Generator（自动支持 ReAct）
playbook = LangChainPlaybook()
result = await generator.generate(
    question="What's (3 + 5) x 12?",
    context="",
    playbook=playbook,
    enable_tools=True,  # 启用工具使用
)

print(f"Answer: {result.final_answer}")
if result.tool_calls:
    print(f"Tools used: {len(result.tool_calls)}")
```

### 直接使用 MultiServerMCPClient

如果你更喜欢直接使用官方 API：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from ace_langchain import LangChainGenerator, LangChainPlaybook

# 1. 创建 MultiServerMCPClient
client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"],
    }
})

# 2. 获取工具
tools = await client.get_tools()

# 3. 使用工具
model = ChatOpenAI(model="gpt-4o-mini")
generator = LangChainGenerator(model=model, tools=tools)

playbook = LangChainPlaybook()
result = await generator.generate(
    question="What's (3 + 5) x 12?",
    context="",
    playbook=playbook,
    enable_tools=True,
)
```

## 📋 服务器配置选项

### stdio 传输（本地子进程）

```python
{
    "server_name": {
        "transport": "stdio",
        "command": "python",  # 或 "node", "bash" 等
        "args": ["/absolute/path/to/server.py"],
    }
}
```

### streamable_http 传输（HTTP 服务器）

```python
{
    "server_name": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",
        "headers": {  # 可选
            "Authorization": "Bearer token",
        },
    }
}
```

### sse 传输（Server-Sent Events）

```python
{
    "server_name": {
        "transport": "sse",
        "url": "http://localhost:8000/mcp",
    }
}
```

## ✅ 优势

- ✅ **简单** - 一个函数调用即可获取工具
- ✅ **官方支持** - 使用 LangChain 官方推荐的方式
- ✅ **无需自定义客户端** - 不需要 `mcp_client.py`
- ✅ **自动 ReAct** - Generator 自动支持工具调用
- ✅ **多服务器支持** - 可以同时连接多个 MCP 服务器

## 📚 参考

- [LangChain MCP 文档](https://docs.langchain.com/oss/python/langchain/mcp)
- [MCP 官方文档](https://modelcontextprotocol.io/)

