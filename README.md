# LangChain演示项目

这个项目展示了LangChain框架的基本用法和核心概念，帮助开发者快速入门和理解LangChain的强大功能。

## 项目特点

- 支持多种大语言模型（OpenAI、DeepSeek、本地Ollama）
- 交互式菜单，易于使用和学习
- 包含丰富的示例代码，涵盖LangChain核心概念
- 模块化设计，便于扩展和定制
- 使用uv进行依赖管理，提供更快的包安装和运行体验

## 项目结构

```
LangchainDemo/
├── examples/                # 各种功能演示
│   ├── __init__.py          # 包初始化文件
│   ├── models.py            # 模型配置和选择
│   ├── chat_models.py       # 聊天模型示例
│   ├── chains.py            # 链示例
│   ├── memory.py            # 记忆示例
│   └── agents.py            # 代理示例
├── main.py                  # 主程序入口
├── .env                     # 环境变量配置
├── pyproject.toml           # 项目依赖配置
└── README.md                # 项目说明
```

## 环境要求

- Python 3.10+
- uv包管理工具
- 可选：OpenAI API密钥
- 可选：DeepSeek API密钥
- 可选：本地Ollama服务

## 安装

1. 安装uv（如果尚未安装）：

```bash
curl -sSf https://install.python-poetry.org | python3 -
```

2. 克隆仓库：

```bash
git clone https://github.com/csc-0812/LangchainDemo.git
cd LangchainDemo
```

3. 使用uv安装依赖：

```bash
uv pip install -e .
```

4. 配置环境变量：

复制`.env文件（如果存在）或创建新的`.env`文件，并设置您的API密钥：

```
# 模型类型: openai, deepseek, ollama
MODEL_TYPE=ollama
# API地址
API_BASE=http://localhost:11434
# API密钥
API_KEY=your_api_key_here
# 模型名称
MODEL_NAME=deepseek-r1:7b
# 模型温度
TEMPERATURE=0.7
# 最大回复长度
MAX_TOKENS=1000
# 最大回复次数  
TOP_P=1
```

## 使用方法

### 使用uv运行（推荐）

使用uv运行主程序，通过交互式菜单选择示例：

```bash
uv run main.py
```

### 命令行参数

您也可以使用命令行参数直接运行特定示例：

```bash
# 使用特定模型运行示例
uv run main.py --model ollama --example 1

# 指定模型名称
uv run main.py --model ollama --name llama2 --example 2
```

### 直接运行示例模块

您还可以直接运行特定的示例模块：

```bash
uv run -m examples.chat_models
uv run -m examples.chains
uv run -m examples.memory
uv run -m examples.agents
```

### 使用Python直接运行（替代方案）

如果您更习惯使用Python直接运行，也可以：

```bash
python main.py
# 或
python -m examples.chat_models
```

## 功能演示

本项目包含以下LangChain核心概念的演示：

### 1. 聊天模型 (Chat Models)

- 基本聊天模型使用
- 系统消息和角色设定
- 提示模板的使用

### 2. 链 (Chains)

- 简单链示例
- 顺序链示例
- JSON输出链示例

### 3. 记忆 (Memory)

- 对话缓冲记忆
- 对话摘要记忆

### 4. 代理 (Agents)

- 基本代理示例
- 检索增强代理示例

## 模型支持

项目支持以下模型类型：

- **OpenAI**: 需要有效的API密钥
- **DeepSeek**: 需要有效的API密钥和API地址
- **Ollama**: 需要本地运行Ollama服务 (http://localhost:11434)

## 依赖

- langchain: LangChain核心库
- langchain-openai: OpenAI模型集成
- langchain-community: 社区模型和工具集成
- python-dotenv: 环境变量管理
- langchain-ollama: Ollama模型集成

## 贡献

欢迎提交问题和拉取请求！

## 许可

MIT
