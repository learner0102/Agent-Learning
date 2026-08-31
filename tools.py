# tools.py - Native Tool Calling 重构版
import functools
from typing import List
from langchain_core.tools import tool
from knowledge_base import KnowledgeBase
from config import logger


def create_search_tool(kb: KnowledgeBase):
    """
    使用闭包将 KnowledgeBase 绑定到工具函数，避免全局变量。
    每次 init_tools() 创建独立闭包，多线程安全。
    """
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        从知识库中检索相关信息。
        当用户需要基于已有知识库回答问题、查找特定主题内容时，使用此工具。
        输入：检索查询字符串（简洁明确的问题）
        返回：相关文档片段及其关联度信息
        """
        try:
            contents, scores = kb.retrieve(query, top_k=3)
            if not contents:
                return "知识库中未找到相关信息"
            
            result = kb.format_retrieved_context(contents, scores)
            return result
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return f"检索过程出错: {str(e)}"
    
    return search_knowledge_base


# 向后兼容：初始化时更新全局 _search_tool 引用
_global_tool = None


def init_tools(knowledge_base: KnowledgeBase):
    """初始化工具（注入知识库依赖）"""
    global _global_tool
    _global_tool = create_search_tool(knowledge_base)


def get_tools():
    """获取所有可用工具列表"""
    return [_global_tool]
