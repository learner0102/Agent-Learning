# knowledge_base.py
import time
from typing import List, Tuple, Optional, Dict
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import logger, API_KEY, BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL

# 导入外部markdown分块工具
from chunks import _split_paragraphs_with_headings, _chunk_paragraphs


class KnowledgeBase:
    """
    RAG知识库管理类
    支持两种分块模式：
        1. 传统字符递归分割（默认）
        2. Markdown标题感知 + Token智能分块（use_heading_chunk=True）
    新增：分批向量化构建FAISS，带容错、进度打印，避免大文件爆显存
    """
    def __init__(
            self,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = CHUNK_OVERLAP,
            use_heading_chunk: bool = False,
            chunk_tokens: int = 512,
            overlap_tokens: int = 100,
            batch_size: int = 32,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_heading_chunk = use_heading_chunk
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.batch_size = batch_size
        self.vectorstore: Optional[FAISS] = None
        self.doc_count = 0

        # 初始化Embedding模型（本地Qwen3‑Embedding‑0.6B）
        self.embeddings = HuggingFaceEmbeddings(
            model_name=r".\study_line\eni\models",
            model_kwargs={"device":"cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # 传统文本分割器（字符模式）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        logger.info(
            f"知识库初始化: chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"use_heading_chunk={use_heading_chunk}, chunk_tokens={chunk_tokens}, batch_size={batch_size}"
        )

    def _build_vector_store_batch(self, documents: List[Document]) -> FAISS:
        """✅新增：分批构建FAISS向量库，复刻index_chunks的分批+容错逻辑"""
        if not documents:
            raise ValueError("没有待索引的文档块")

        all_docs = []
        total = len(documents)
        logger.info(f"[RAG] 开始分批向量化，总块数:{total}, batch_size:{self.batch_size}")

        faiss_store: Optional[FAISS] = None

        for start_idx in range(0, total, self.batch_size):
            batch_docs = documents[start_idx: start_idx + self.batch_size]
            try:
                # 分批处理这一批
                if faiss_store is None:
                    # 第一批：初始化FAISS库
                    faiss_store = FAISS.from_documents(batch_docs, self.embeddings)
                else:
                    # 后续批次：add_documents追加到已有向量库
                    faiss_store.add_documents(batch_docs)

                processed = min(start_idx + self.batch_size, total)
                logger.info(f"[RAG] Embedding进度: {processed}/{total}")

            except Exception as e:
                # 单一批次失败，打印警告，跳过该批次，不整体崩溃（容错逻辑）
                logger.warning(f"[WARNING] 批次 {start_idx} 编码失败，跳过该批，error:{str(e)}")

        if faiss_store is None:
            raise RuntimeError("所有批次全部处理失败，无法构建向量库")

        return faiss_store

    def load_and_index(self, file_path: str) -> int:
        """加载文本文件并构建向量索引，支持两种分块策略"""
        logger.info(f"开始加载知识库: {file_path}")
        start_time = time.time()
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if self.use_heading_chunk:
            logger.info("使用【Markdown标题感知‑Token分块】策略")
            paragraphs = _split_paragraphs_with_headings(raw_text)
            chunk_dicts = _chunk_paragraphs(paragraphs, self.chunk_tokens, self.overlap_tokens)
            documents = [
                Document(
                    page_content=item["content"],
                    metadata={
                        "chunk_id": idx,
                        "heading_path": item["heading_path"],
                        "start_offset": item["start"],
                        "end_offset": item["end"]
                    }
                )
                for idx, item in enumerate(chunk_dicts)
            ]
        else:
            logger.info("使用【传统字符递归分割】策略")
            lines = raw_text.splitlines()
            raw_texts = [line.strip() for line in lines if line.strip()]
            raw_text_join = "\n\n".join(raw_texts)
            chunks = self.text_splitter.split_text(raw_text_join)
            documents = [
                Document(page_content=chunk, metadata={"chunk_id": i})
                for i, chunk in enumerate(chunks)
            ]

        self.doc_count = len(documents)
        logger.info(f"分块完成: {len(documents)} 个文档块")

        # ========== 修改点：替换原来 self.vectorstore = FAISS.from_documents(documents, self.embeddings) ==========
        self.vectorstore = self._build_vector_store_batch(documents)

        elapsed = time.time() - start_time
        logger.info(f"索引构建完成，耗时: {elapsed:.2f}s")
        return len(documents)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Tuple[List[str], List[float]]:
        """检索最相关的文档片段【原有接口保持不变】"""
        contents, scores, _ = self.retrieve_with_meta(query, top_k)
        return contents, scores

    def retrieve_with_meta(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Tuple[List[str], List[float], List[Dict]]:
        """检索，返回【内容，分数，完整元数据】推荐上层使用这个"""
        if self.vectorstore is None:
            raise ValueError("知识库未初始化，请先调用 load_and_index()")
        logger.info(f"检索: {query[:80]}... (top_k={top_k})")
        start_time = time.time()
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        contents = [doc.page_content for doc, _ in docs_with_scores]
        scores = [float(score) for _, score in docs_with_scores]
        metas = [doc.metadata for doc, _ in docs_with_scores]

        for i, (content, score) in enumerate(zip(contents, scores)):
            logger.debug(f"  片段{i+1}: 相似度={score:.4f}, 长度={len(content)}, meta={metas[i]}")

        elapsed = time.time() - start_time
        logger.info(f"检索完成，召回 {len(contents)} 个片段，耗时: {elapsed:.3f}s")
        return contents, scores, metas


    def format_retrieved_context(self, contents: List[str], scores: List[float]) -> str:
        """格式化检索结果为Prompt输入"""
        formatted = []
        for i, (content, score) in enumerate(zip(contents, scores)):
            confidence = "高" if score > 0.7 else ("中" if score > 0.5 else "低")
            formatted.append(
                f"【片段{i+1}】(相关度: {confidence}, 分数: {score:.3f})\n{content}"
            )
        return "\n\n".join(formatted)

    # ====== 额外附赠：FAISS保存/加载磁盘方法，非常实用 ======
    def save_faiss(self, save_dir: str):
        """把FAISS向量库保存到本地磁盘"""
        if self.vectorstore is None:
            raise ValueError("向量库尚未构建")
        self.vectorstore.save_local(save_dir)
        logger.info(f"FAISS向量库已保存至 {save_dir}")

    def load_faiss(self, load_dir: str):
        """从磁盘加载FAISS向量库，不用重新embedding"""
        self.vectorstore = FAISS.load_local(
            load_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"FAISS向量库从 {load_dir} 加载完成")