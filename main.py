"""
LangChain演示项目主程序

这个程序是LangChain演示项目的入口点，允许用户选择运行不同的示例和模型。
支持OpenAI、DeepSeek和本地Ollama模型，并提供交互式菜单和命令行参数。
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from examples.models import get_model_info, ModelType

# 加载环境变量
load_dotenv()

# 全局变量
SELECTED_MODEL_TYPE: ModelType = os.getenv("MODEL_TYPE", "ollama")  # 默认使用Ollama
SELECTED_MODEL_NAME = os.getenv("MODEL_NAME")  # 默认使用模型类型的默认模型

def check_model_availability():
    """检查所选模型是否可用"""
    model_info = get_model_info()
    
    if not model_info[SELECTED_MODEL_TYPE]["available"]:
        if SELECTED_MODEL_TYPE == "openai":
            print(f"错误: OpenAI模型不可用。请在.env文件中设置有效的API_KEY。")
        elif SELECTED_MODEL_TYPE == "deepseek":
            print(f"错误: DeepSeek模型不可用。请在.env文件中设置有效的API_KEY和API_BASE。")
        elif SELECTED_MODEL_TYPE == "ollama":
            print(f"错误: Ollama模型不可用。请确保Ollama服务正在运行(http://localhost:11434)。")
        return False
    
    if SELECTED_MODEL_NAME and SELECTED_MODEL_NAME not in model_info[SELECTED_MODEL_TYPE]["models"]:
        print(f"警告: 指定的模型 '{SELECTED_MODEL_NAME}' 可能不可用。")
        print(f"可用的{SELECTED_MODEL_TYPE}模型: {', '.join(model_info[SELECTED_MODEL_TYPE]['models'])}")
        confirm = input("是否继续? (y/n): ")
        if confirm.lower() != 'y':
            return False
    
    return True

def display_model_info():
    """显示当前选择的模型信息"""
    print(f"\n当前使用的模型: {SELECTED_MODEL_TYPE}" + 
          (f" ({SELECTED_MODEL_NAME})" if SELECTED_MODEL_NAME else ""))

def display_menu():
    """显示菜单选项"""
    print("\n" + "="*50)
    print("LangChain演示项目".center(50))
    print("="*50)
    display_model_info()
    print("\n请选择要运行的示例:")
    print("1. 聊天模型示例")
    print("2. 链示例")
    print("3. 记忆示例")
    print("4. 代理示例")
    print("5. 模型选择")
    print("0. 退出")
    print("-"*50)

def display_model_menu():
    """显示模型选择菜单"""
    model_info = get_model_info()
    
    print("\n" + "="*50)
    print("模型选择".center(50))
    print("="*50)
    
    print("\n请选择模型类型:")
    print("1. OpenAI" + (" (可用)" if model_info["openai"]["available"] else " (不可用)"))
    print("2. DeepSeek" + (" (可用)" if model_info["deepseek"]["available"] else " (不可用)"))
    print("3. Ollama" + (" (可用)" if model_info["ollama"]["available"] else " (不可用)"))
    print("0. 返回主菜单")
    print("-"*50)

def select_model():
    """选择模型类型和名称"""
    global SELECTED_MODEL_TYPE, SELECTED_MODEL_NAME
    
    while True:
        display_model_menu()
        choice = input("\n请输入您的选择 (0-3): ")
        
        if choice == "0":
            return
        elif choice == "1":
            SELECTED_MODEL_TYPE = "openai"
            select_specific_model("openai")
            return
        elif choice == "2":
            SELECTED_MODEL_TYPE = "deepseek"
            select_specific_model("deepseek")
            return
        elif choice == "3":
            SELECTED_MODEL_TYPE = "ollama"
            select_specific_model("ollama")
            return
        else:
            print("无效的选择，请重试。")

def select_specific_model(model_type: ModelType):
    """选择特定模型类型下的具体模型"""
    global SELECTED_MODEL_NAME
    
    model_info = get_model_info()
    if not model_info[model_type]["available"]:
        print(f"{model_type}模型不可用，请检查配置。")
        return
    
    models = model_info[model_type]["models"]
    if not models:
        print(f"未找到可用的{model_type}模型。")
        return
    
    print(f"\n可用的{model_type}模型:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    print("0. 使用默认模型")
    
    while True:
        try:
            choice = input("\n请选择模型 (0 表示默认): ")
            if choice == "0":
                SELECTED_MODEL_NAME = None
                print(f"已选择默认{model_type}模型")
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                SELECTED_MODEL_NAME = models[idx]
                print(f"已选择模型: {SELECTED_MODEL_NAME}")
                break
            else:
                print("无效的选择，请重试。")
        except ValueError:
            print("请输入有效的数字。")

def run_example(choice):
    """运行选择的示例"""
    # 创建模型参数
    model_kwargs = {
        "model_type": SELECTED_MODEL_TYPE,
        "model_name": SELECTED_MODEL_NAME
    }
    
    try:
        if choice == "1":
            print("\n运行聊天模型示例...")
            from examples import chat_models
            chat_models.basic_chat_example()
            chat_models.chat_with_system_message()
            chat_models.chat_with_prompt_template()
        elif choice == "2":
            print("\n运行链示例...")
            from examples import chains
            chains.simple_chain_example()
            chains.sequential_chain_example()
            chains.json_output_chain_example()
        elif choice == "3":
            print("\n运行记忆示例...")
            from examples import memory
            memory.conversation_buffer_memory_example()
            memory.conversation_summary_memory_example()
        elif choice == "4":
            print("\n运行代理示例...")
            from examples import agents
            agents.basic_agent_example()
            agents.retrieval_agent_example()
        elif choice == "5":
            select_model()
        else:
            print("无效的选择，请重试。")
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        print("请检查模型配置和网络连接。")

def parse_arguments():
    """解析命令行参数"""
    global SELECTED_MODEL_TYPE, SELECTED_MODEL_NAME
    
    parser = argparse.ArgumentParser(description="LangChain演示项目")
    parser.add_argument("--model", "-m", choices=["openai", "deepseek", "ollama"], 
                        help="选择模型类型: openai, deepseek, ollama")
    parser.add_argument("--name", "-n", help="指定模型名称")
    parser.add_argument("--example", "-e", type=int, choices=[1, 2, 3, 4],
                        help="直接运行指定示例: 1=聊天模型, 2=链, 3=记忆, 4=代理")
    
    args = parser.parse_args()
    
    if args.model:
        SELECTED_MODEL_TYPE = args.model
    
    if args.name:
        SELECTED_MODEL_NAME = args.name
    
    return args

def main():
    """主函数"""
    print("欢迎使用LangChain演示项目!")
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查模型可用性
    if not check_model_availability():
        return
    
    # 如果指定了示例，直接运行
    if args.example:
        run_example(str(args.example))
        return
    
    # 交互式菜单
    while True:
        display_menu()
        choice = input("\n请输入您的选择 (0-5): ")
        
        if choice == "0":
            print("\n感谢使用LangChain演示项目，再见!")
            break
        
        run_example(choice)
        
        input("\n按Enter键继续...")

if __name__ == "__main__":
    main()