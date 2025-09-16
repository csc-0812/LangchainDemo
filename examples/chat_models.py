"""
LangChain聊天模型示例

这个示例展示了如何使用LangChain与各种聊天模型进行交互，包括OpenAI、DeepSeek和Ollama模型。
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 导入模型工具
from .models import get_chat_model

def basic_chat_example(model_kwargs: Optional[Dict[str, Any]] = None):
    """
    基本聊天模型示例
    
    Args:
        model_kwargs: 可选的模型参数，包括model_type和model_name
    """
    print("\n=== 基本聊天模型示例 ===")
    
    # 初始化聊天模型
    chat = get_chat_model(model_kwargs)

    # 发送单个消息
    message = HumanMessage(content="用简单的术语解释量子计算")
    response = chat.invoke([message])
    
    print(f"问题: {message.content}")
    print(f"回答: {response.content}")
    print()

def chat_with_system_message(model_kwargs: Optional[Dict[str, Any]] = None):
    """
    使用系统消息的聊天示例
    
    Args:
        model_kwargs: 可选的模型参数，包括model_type和model_name
    """
    print("\n=== 使用系统消息的聊天示例 ===")
    
    # 初始化聊天模型
    chat = get_chat_model(model_kwargs)

    # 创建消息列表，包括系统消息
    messages = [
        SystemMessage(content="你是一位友好的AI助手，专注于用简单的语言解释复杂的科学概念。"),
        HumanMessage(content="解释一下黑洞是什么，就像我是一个10岁的孩子。")
    ]
    
    # 发送消息
    response = chat.invoke(messages)
    
    print(f"系统: {messages[0].content}")
    print(f"问题: {messages[1].content}")
    print(f"回答: {response.content}")
    print()

def chat_with_prompt_template(model_kwargs: Optional[Dict[str, Any]] = None):
    """
    使用提示模板的聊天示例
    
    Args:
        model_kwargs: 可选的模型参数，包括model_type和model_name
    """
    print("\n=== 使用提示模板的聊天示例 ===")

    # 创建聊天提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专家{role}。你的任务是{task}。"),
        ("human", "{input}")
    ])
    
    # 初始化聊天模型
    chat = get_chat_model(model_kwargs)
    
    # 创建链
    chain = prompt | chat
    
    # 运行链
    response = chain.invoke({
        "role": "历史学家",
        "task": "用生动有趣的方式解释历史事件",
        "input": "简单介绍一下丝绸之路的历史和重要性"
    })
    
    print("提示模板填充后:")
    print("系统: 你是一位专家历史学家。你的任务是用生动有趣的方式解释历史事件。")
    print("问题: 简单介绍一下丝绸之路的历史和重要性")
    print(f"回答: {response.content}")

if __name__ == "__main__":
    # 如果直接运行此文件，使用默认模型
    basic_chat_example()
    chat_with_system_message()
    chat_with_prompt_template()