"""
LangChain模型配置模块

这个模块提供了不同LLM模型的配置和选择功能，支持OpenAI、DeepSeek和本地Ollama模型。
"""

import os
import requests
from typing import Dict, Any, Literal, Optional, List, TypedDict
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# 定义模型类型
ModelType = Literal["openai", "deepseek", "ollama"]

# 加载环境变量
load_dotenv()

def get_chat_model(model_kwargs: Optional[Dict[str, Any]] = None) -> BaseChatModel:
    """
    获取聊天模型实例
    
    Args:
        model_kwargs: 可选的模型参数，包括model_type和model_name
        
    Returns:
        BaseChatModel: 聊天模型实例
    """
    # 获取默认配置
    config = {}
    
    # 从环境变量获取配置
    config["model_type"] = os.getenv("MODEL_TYPE")
    config["api_base"] = os.getenv("API_BASE")
    config["api_key"] = os.getenv("API_KEY")
    config["model"] = os.getenv("MODEL_NAME")
    config["temperature"] = float(os.getenv("TEMPERATURE", 0.7))
    config["max_tokens"] = int(os.getenv("MAX_TOKENS", 1000))
    config["top_p"] = float(os.getenv("TOP_P", 1.0))
    
    # 如果提供了model_kwargs，则更新配置
    if model_kwargs:
        if model_kwargs.get("model_type"):
            config["model_type"] = model_kwargs["model_type"]
        if model_kwargs.get("model_name"):
            config["model"] = model_kwargs["model_name"]
    
    # 根据模型类型创建相应的模型实例
    if config["model_type"] == "ollama":
        return ChatOllama(
            base_url=config["api_base"],
            model=config["model"],
            temperature=config["temperature"]
        )
    else:
        return ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["api_base"],
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"]
        )

def get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    获取可用模型信息
    
    Returns:
        Dict: 包含各模型类型可用性和模型列表的字典
    """
    model_info = {
        "openai": {
            "available": False,
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        },
        "deepseek": {
            "available": False,
            "models": ["deepseek-chat", "deepseek-coder"]
        },
        "ollama": {
            "available": False,
            "models": []
        }
    }
    
    # 检查OpenAI可用性
    api_key = os.getenv("API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        model_info["openai"]["available"] = True
    
    # 检查DeepSeek可用性
    api_base = os.getenv("API_BASE")
    if api_key and api_base and "deepseek" in api_base:
        model_info["deepseek"]["available"] = True
    
    # 检查Ollama可用性
    ollama_url = os.getenv("API_BASE", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            model_info["ollama"]["available"] = True
            # 获取Ollama可用模型列表
            models_data = response.json()
            if "models" in models_data:
                model_info["ollama"]["models"] = [model["name"] for model in models_data["models"]]
            else:
                model_info["ollama"]["models"] = ["llama2", "mistral", "gemma", "deepseek-r1"]
    except:
        # 如果无法连接到Ollama服务，则使用默认模型列表
        model_info["ollama"]["models"] = ["llama2", "mistral", "gemma", "deepseek-r1"]
    
    return model_info