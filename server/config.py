import os

def set_environment():
    # OpenAI
    os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
    os.environ['OPENAI_API_KEY'] = "sk-z1iOi2WcPFt3zeXEj0ZNOF6xR4yUI4T4JdxGoBPnblX6c1vn"
    
    # Langchain
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_API_KEY"] = "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"
    
    # Search Engine
    # os.environ["TAVILY_API_KEY"] = "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"
