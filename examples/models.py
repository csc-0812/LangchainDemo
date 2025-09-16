"""
LangChain模型配置模块

这个模块提供了不同LLM模型的配置和选择功能，支持OpenAI、DeepSeek和本地Ollama模型。
"""

import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI  # 确保ChatOpenAI已导入

# 加载环境变量
load_dotenv()

def get_chat_model() -> BaseChatModel:

    # 获取默认配置
    config = {}
    
    config["model_type"] = os.getenv("MODEL_TYPE")
    config["api_base"] = os.getenv("API_BASE")
    config["api_key"] = os.getenv("API_KEY")
    config["model"] = os.getenv("MODEL_NAME")
    config["temperature"] = os.getenv("TEMPERATURE")
    config["max_tokens"] = os.getenv("MAX_TOKENS")
    config["top_p"] = os.getenv("TOP_P")

    # 根据模型类型创建相应的模型实例
    if config["model_type"] == "ollama":
        return ChatOllama(**config)
    else:
        return ChatOpenAI(**config)