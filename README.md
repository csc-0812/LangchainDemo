# LangChain演示项目

这个项目展示了LangChain框架的基本用法和核心概念。

## 项目结构

```
LangchainDemo/
├── examples/                # 各种功能演示
│   ├── chat_models.py       # 聊天模型示例
│   ├── chains.py            # 链示例
│   ├── agents.py            # 代理示例
│   └── memory.py            # 记忆示例
├── main.py                  # 主程序入口
├── .env                     # 环境变量配置
└── README.md                # 项目说明
```

## 安装

使用uv安装依赖：

```bash
uv pip install -e .
```

## 使用方法

1. 在`.env`文件中设置您的OpenAI API密钥
2. 运行主程序：

```bash
python main.py
```

或者运行特定示例：

```bash
python -m examples.chat_models
```

## 功能演示

本项目包含以下LangChain核心概念的演示：

1. **模型 (Models)** - 与各种LLM的集成
2. **提示 (Prompts)** - 提示模板和提示工程
3. **链 (Chains)** - 将多个组件链接在一起
4. **记忆 (Memory)** - 在链的调用之间保持状态
5. **代理 (Agents)** - 让LLM决定使用哪些工具

## 依赖

- langchain
- langchain-openai
- langchain-community
- python-dotenv