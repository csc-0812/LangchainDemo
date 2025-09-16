"""
LangChain代理(Agents)示例

这个示例展示了如何使用LangChain中的代理功能，让语言模型能够使用工具来解决问题。
支持OpenAI、DeepSeek和本地Ollama模型。
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

# 导入模型工具
from .models import get_chat_model

def basic_agent_example():
    """基本代理示例"""
    print("\n=== 基本代理示例 ===")

    # 定义工具函数
    @tool
    def search_weather(location: str) -> str:
        """搜索指定位置的天气信息"""
        weather_data = {
            "北京": "晴朗，温度25°C，湿度45%",
            "上海": "多云，温度28°C，湿度60%",
            "广州": "小雨，温度30°C，湿度75%",
            "深圳": "阵雨，温度29°C，湿度70%",
            "杭州": "晴朗，温度26°C，湿度50%"
        }
        return weather_data.get(location, f"没有找到{location}的天气信息")
    
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式的结果"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    # 创建代理
    tools = [search_weather, calculate]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用的AI助手，可以使用提供的工具来回答用户的问题。
        
        可用工具:
        - search_weather: 搜索指定位置的天气信息
        - calculate: 计算数学表达式的结果
        
        使用工具时，请遵循以下格式:
        思考: 我需要使用什么工具来回答这个问题？
        行动: 工具名称
        行动输入: 工具的输入参数
        观察: 工具的输出结果
        
        当你有了足够的信息来回答用户的问题时，直接提供答案，不需要使用工具。
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建模型
    llm = get_chat_model()
    
    # 创建代理
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 创建代理执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 测试代理
    questions = [
        "北京今天的天气怎么样？",
        "计算一下123乘以456是多少？",
        "上海和广州哪个地方更热？"
    ]
    
    chat_history = []
    
    for question in questions:
        print(f"\n用户: {question}")
        response = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })
        print(f"AI: {response['output']}")
        
        # 更新对话历史
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response["output"]))

def retrieval_agent_example():
    """检索增强代理示例"""
    print("\n=== 检索增强代理示例 ===")

    # 创建一些示例文档
    documents = [
        Document(page_content="Python是一种高级编程语言，以其简洁、易读的语法而闻名。它支持多种编程范式，包括面向对象、命令式和函数式编程。Python由Guido van Rossum创建，于1991年首次发布。", metadata={"source": "python_info.txt"}),
        Document(page_content="JavaScript是一种脚本语言，主要用于Web开发。它可以使网页具有交互性，是现代Web开发的核心技术之一。JavaScript最初由Netscape的Brendan Eich开发。", metadata={"source": "javascript_info.txt"}),
        Document(page_content="机器学习是人工智能的一个子领域，它使计算机系统能够从数据中学习和改进，而无需明确编程。常见的机器学习算法包括线性回归、决策树、随机森林和神经网络。", metadata={"source": "ml_info.txt"}),
        Document(page_content="深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。", metadata={"source": "dl_info.txt"}),
    ]
    
    # 创建向量存储
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        "search_documents",
        "搜索文档库中与查询相关的信息"
    )
    
    # 定义计算工具
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式的结果"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    # 创建代理
    tools = [retriever_tool, calculate]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用的AI助手，可以使用提供的工具来回答用户的问题。
        
        可用工具:
        - search_documents: 搜索文档库中与查询相关的信息
        - calculate: 计算数学表达式的结果
        
        当用户询问的信息可能在文档库中时，请使用search_documents工具。
        当需要进行数学计算时，请使用calculate工具。
        
        如果你不确定答案，请诚实地说出来，不要编造信息。
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建模型
    llm = get_chat_model()
    
    # 创建代理
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 创建代理执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 测试代理
    questions = [
        "Python语言是谁创建的？",
        "深度学习和机器学习有什么区别？",
        "如果我有5个Python项目和3个JavaScript项目，总共有多少个项目？"
    ]
    
    chat_history = []
    
    for question in questions:
        print(f"\n用户: {question}")
        response = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })
        print(f"AI: {response['output']}")
        
        # 更新对话历史
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response["output"]))

if __name__ == "__main__":
    # 如果直接运行此文件，使用默认模型
    basic_agent_example()
    retrieval_agent_example()