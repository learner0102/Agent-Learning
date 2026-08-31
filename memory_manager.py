# memory_manager.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from config import logger
import re  # 加到文件顶部
from langchain_community.vectorstores import FAISS


class MemoryManager:
    def __init__(self, user_id: str, memory_dir: str = "study_line\\eni\\memory_store",
                 embeddings=None):
        self.user_id = user_id
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        self.disk_file = os.path.join(self.memory_dir, f"mem_{user_id}.json")
        self.embeddings = embeddings          # ← 复用知识库的 embedding，避免重复加载模型

        self.short_term: List[Dict[str, Any]] = []
        self.long_term: List[Dict[str, Any]] = []
        self._load_from_disk()
        self.PERSIST_THRESHOLD = 0.7

        self.vectorstore = None               # 语义索引
        self._rebuild_index()                 # 启动时用已加载的记忆建索引

    def _rebuild_index(self):
        """用当前 long_term 重建 FAISS 索引。记忆量小，全量重建成本可忽略。"""
        self.vectorstore = None
        if self.embeddings is None or not self.long_term:
            return
        docs = [
            Document(page_content=m["content"], metadata={"mem_idx": i})
            for i, m in enumerate(self.long_term)
        ]
        try:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        except Exception as e:
            logger.warning(f"[Memory] 语义索引重建失败，将回退关键词检索: {e}")

    def get_session_context(self, limit: int = 6) -> str:
        """将最近的短期记忆格式化为会话历史字符串"""
        items = self.short_term[-limit:]
        if not items:
            return ""
        return "\n".join([f"- {item['content']}" for item in items])

    def _load_from_disk(self):
        """读取磁盘上的重要长期记忆"""
        if os.path.exists(self.disk_file):
            try:
                with open(self.disk_file, "r", encoding="utf-8") as f:
                    self.long_term = json.load(f)
                logger.info(f"[Memory] 用户 {self.user_id} 加载长期记忆 {len(self.long_term)} 条")
            except Exception as e:
                logger.warning(f"[Memory] 读取记忆文件失败:{e}")
                self.long_term = []
        else:
            self.long_term = []

    def _save_to_disk(self):
        """仅保存 long_term（重要记忆）到本地"""
        try:
            with open(self.disk_file, "w", encoding="utf-8") as f:
                json.dump(self.long_term, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"[Memory] 保存长期记忆失败: {e}")

    def add(self, content, memory_type, session_id, importance=0.5,
            event_type=None, concept=None):
        item = {
            "content": content, "memory_type": memory_type,
            "session_id": session_id, "importance": importance,
            "event_type": event_type, "concept": concept,
            "create_time": datetime.now().isoformat()
        }
        if memory_type == "working":
            self.short_term.append(item)
        elif memory_type in ("episodic", "semantic"):
            self.long_term.append(item)
            if importance >= self.PERSIST_THRESHOLD:
                self._save_to_disk()
            self._rebuild_index()             # ← 新增记忆后更新语义索引
        else:
            raise ValueError(f"不支持memory_type: {memory_type}")

    @staticmethod
    def _tokenize(text: str) -> set:
        """关键词兜底用的分词：英文按单词、中文按相邻二字 bigram。"""
        text = text.lower()
        tokens = set(re.findall(r"[a-z0-9]+", text))
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        for i in range(len(cjk) - 1):
            tokens.add(cjk[i] + cjk[i + 1])
        return tokens


    def search(self, query: str, limit: int = 5) -> str:
        # 1) 语义召回（优先）
        if self.vectorstore is not None:
            try:
                docs = self.vectorstore.similarity_search(query, k=limit)
                if docs:
                    lines = []
                    for idx, doc in enumerate(docs):
                        m = self.long_term[doc.metadata["mem_idx"]]
                        lines.append(
                            f"{idx+1}. [{m['memory_type']}] {m['content']} "
                            f"(重要度:{m['importance']})"
                        )
                    return "\n".join(lines)
            except Exception as e:
                logger.warning(f"[Memory] 语义检索失败，回退关键词: {e}")

        # 2) 关键词召回兜底（语义索引不可用时降级）
        q = query.lower()
        q_terms = self._tokenize(query)
        scored = []
        for m in self.long_term:
            content = m["content"].lower()
            score = 0
            for t in q_terms:
                if t in content:
                    score += 1
            if q in content:
                score += 10
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: (x[0], x[1].get("importance", 0)), reverse=True)
        selected = scored[:limit]
        if not selected:
            return "没有找到相关长期记忆。"
        return "\n".join(
            f"{idx+1}. [{m['memory_type']}] {m['content']} (重要度:{m['importance']})"
            for idx, (_, m) in enumerate(selected)
        )

    
    def get_short_term(self) -> List[Dict[str, Any]]:
        """获取当前会话全部完整短期对话（内存）"""
        return self.short_term

    def clear_short_term(self):
        """清空当前会话短期记忆"""
        self.short_term.clear()

    def summary(self, limit: int = 10) -> str:
        """获取重要度最高的长期记忆摘要"""
        sorted_mem = sorted(self.long_term, key=lambda x:x.get("importance",0), reverse=True)[:limit]
        if not sorted_mem:
            return "暂无长期记忆"
        return "\n".join([f"- {m['content']}" for m in sorted_mem])

    def get_long_all(self):
        return self.long_term
