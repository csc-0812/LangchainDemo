"""
LangChain记忆(Memory)示例

这个示例展示了如何在LangChain中使用不同类型的记忆组件来保持对话状态。
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage

# 导入模型工具
from .models import get_chat_model

def conversation_buffer_memory_example():
    """对话缓冲记忆示例"""
    print("\n=== 对话缓冲记忆示例 ===")

    # 创建记忆组件
    memory = ConversationBufferMemory(return_messages=True)
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位友好的AI助手，能够记住对话历史。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 创建模型
    model = get_chat_model()
    
    # 创建对话链
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=model,
        verbose=False
    )
    
    # 模拟对话
    questions = [
        "我叫张明，很高兴认识你。",
        "我最喜欢的颜色是蓝色。",
        "你还记得我的名字吗？",
        "我最喜欢的颜色是什么？"
    ]
    
    for question in questions:
        response = conversation.predict(input=question)
        print(f"用户: {question}")
        print(f"AI: {response}")
        print()
    
    # 显示记忆中存储的内容
    print("记忆中存储的内容:")
    memory_variables = memory.load_memory_variables({})
    for i, message in enumerate(memory_variables["history"]):
        role = "用户" if isinstance(message, HumanMessage) else "AI"
        print(f"{role}: {message.content}")
    print()

def conversation_summary_memory_example():
    """对话摘要记忆示例"""
    print("\n=== 对话摘要记忆示例 ===")

    # 创建模型
    model = get_chat_model()
    
    # 创建记忆组件 - 使用相同的模型进行摘要
    memory = ConversationSummaryMemory(
        llm=model,
        return_messages=True
    )
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位友好的AI助手，能够记住对话历史。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 创建对话链
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=model,
        verbose=False
    )
    
    # 模拟一个较长的对话
    conversation_history = [
        "你好，我是李华。我正在计划一次旅行。",
        "我想去日本旅行，你有什么建议吗？",
        "我特别喜欢历史和美食。",
        "我计划在东京待3天，京都待2天。",
        "我的预算大约是1500美元，不包括机票。"
    ]
    
    print("进行一段对话...")
    for message in conversation_history:
        response = conversation.predict(input=message)
        print(f"用户: {message}")
        print(f"AI: {response}")
        print()
    
    # 测试记忆效果
    test_question = "你能总结一下我的旅行计划吗？"
    response = conversation.predict(input=test_question)
    print(f"用户: {test_question}")
    print(f"AI: {response}")
    print()
    
    # 显示摘要记忆
    print("记忆中的对话摘要:")
    # print(memory.buffer)
    print("Memory对象:", memory)
    print("Memory属性列表:")
    for attr in dir(memory):
        if not attr.startswith('__'):
            try:
                value = getattr(memory, attr)
                if not callable(value):
                    print(f"  - {attr}: {value}")
            except Exception as e:
                print(f"  - {attr}: 无法访问 ({str(e)})")
                
    # 显示记忆变量
    print("Memory变量内容:")
    memory_vars = memory.load_memory_variables({})
    for key, value in memory_vars.items():
        print(f"  - {key}:")
        if isinstance(value, list):
            for item in value:
                print(f"    * {type(item).__name__}: {item}")
        else:
            print(f"    * {value}")
    print()

if __name__ == "__main__":
    # 如果直接运行此文件，使用默认模型
    conversation_buffer_memory_example()
    conversation_summary_memory_example()