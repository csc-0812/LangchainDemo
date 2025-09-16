"""
LangChain链示例

这个示例展示了如何使用LangChain中的链(Chains)概念来组合多个组件。
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 导入模型工具
from .models import get_chat_model

def simple_chain_example(model_kwargs: Optional[Dict[str, Any]] = None):
    """
    简单链示例
    
    Args:
        model_kwargs: 可选的模型参数，包括model_type和model_name
    """
    print("\n=== 简单链示例 ===")

    # 创建提示模板
    prompt = ChatPromptTemplate.from_template(
        "给我{topic}的5个有趣事实，用简洁的语言描述。"
    )
    
    # 创建模型
    model = get_chat_model(model_kwargs)
    
    # 创建输出解析器
    output_parser = StrOutputParser()
    
    # 组合成链
    chain = prompt | model | output_parser
    
    # 运行链
    topic = "古埃及"
    result = chain.invoke({"topic": topic})
    
    print(f"关于{topic}的有趣事实:")
    print(result)
    print()

def sequential_chain_example(model_kwargs: Optional[Dict[str, Any]] = None):
    """
    顺序链示例
    
    Args:
        model_kwargs: 可选的模型参数，包括model_type和model_name
    """
    print("\n=== 顺序链示例 ===")

    # 创建模型
    model = get_chat_model(model_kwargs)
    
    # 第一个链：生成故事主题
    topic_prompt = ChatPromptTemplate.from_template(
        "生成一个有趣的故事主题，包含以下元素：{element1}和{element2}。只返回主题，不要写故事。"
    )
    topic_chain = topic_prompt | model | StrOutputParser()
    
    # 第二个链：根据主题写故事
    story_prompt = ChatPromptTemplate.from_template(
        "根据以下主题写一个简短的故事（不超过200字）：\n\n{topic}"
    )
    story_chain = story_prompt | model | StrOutputParser()
    
    # 运行第一个链
    element1 = "时间旅行"
    element2 = "古代图书馆"
    topic_result = topic_chain.invoke({"element1": element1, "element2": element2})
    
    print(f"生成的故事主题: {topic_result}")
    
    # 运行第二个链
    story_result = story_chain.invoke({"topic": topic_result})
    
    print(f"生成的故事:\n{story_result}")
    print()

def json_output_chain_example(model_kwargs: Optional[Dict[str, Any]] = None):
    """
    JSON输出链示例
    
    Args:
        model_kwargs: 可选的模型参数，包括model_type和model_name
    """
    print("\n=== JSON输出链示例 ===")

    # 定义输出模式
    class MovieRecommendation(BaseModel):
        title: str = Field(description="电影标题")
        director: str = Field(description="导演名称")
        year: int = Field(description="发行年份")
        genre: str = Field(description="电影类型")
        summary: str = Field(description="简短的电影概述")
        reasons: List[str] = Field(description="推荐这部电影的理由列表")
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_template(
        """根据用户的喜好推荐一部电影。
        用户喜好: {preferences}
        
        以JSON格式返回一部电影推荐，包含以下字段:
        - title: 电影标题
        - director: 导演名称
        - year: 发行年份(数字)
        - genre: 电影类型
        - summary: 简短的电影概述
        - reasons: 推荐这部电影的理由列表(至少3个理由)
        """
    )
    
    # 创建模型
    model = get_chat_model(model_kwargs)
    
    # 创建JSON输出解析器
    output_parser = JsonOutputParser(pydantic_object=MovieRecommendation)
    
    # 组合成链
    chain = prompt | model | output_parser
    
    # 运行链
    preferences = "我喜欢科幻电影，特别是那些探索人类与技术关系的电影。我也喜欢有深度的剧情和令人惊讶的结局。"
    try:
        result = chain.invoke({"preferences": preferences})
        
        # 检查结果是否包含所需字段
        required_fields = ["title", "director", "year", "genre", "summary", "reasons"]
        if not all(field in result for field in required_fields):
            raise ValueError("返回结果缺少必要字段")
            
        print("电影推荐结果:")
        print(f"标题: {result['title']}")
        print(f"导演: {result['director']}")
        print(f"年份: {result['year']}")
        print(f"类型: {result['genre']}")
        print(f"概述: {result['summary']}")
        print("推荐理由:")
        for i, reason in enumerate(result['reasons'], 1):
            print(f"  {i}. {reason}")
        print()
    except Exception as e:
        print(f"获取电影推荐时出错: {str(e)}")
        print("原始响应内容:")
        raw_response = model.invoke(prompt.format_prompt(preferences=preferences).to_messages())
        print(raw_response.content)
        print()

if __name__ == "__main__":
    # 如果直接运行此文件，使用默认模型
    simple_chain_example()
    sequential_chain_example()
    json_output_chain_example()