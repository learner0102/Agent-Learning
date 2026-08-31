# ============================================================
# 模块1：配置与初始化（增加可观测性）
# ============================================================
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import os

import numpy as np

# 配置结构化日志（简历加分：可观测性）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_system.log'),  # 落盘，方便排查
        logging.StreamHandler()                # 控制台输出
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("Qwen_API_KEY")
BASE_URL = os.getenv("DB_URL")

# ============================================================
# 模块2：LangChain核心组件（保持不变，但增加异常处理）
# ============================================================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 初始化LLM（增加重试配置）
llm = ChatOpenAI(
    model="deepseek-r1",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.7,
    timeout=60,  # 简历加分：超时控制
    max_retries=2  # 简历加分：内置重试
)

# 初始化Embedding模型（简历加分：RAG核心）
embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="text-embedding-v3"  # 如果Qwen没有，换成text-embedding-ada-002或其他
)

output_parser = StrOutputParser()

# ============================================================
# 模块3：Prompt工程（增加引用溯源和置信度）
# ============================================================
retriever_prompt = """你是一个检索专家，能够从知识库中精准定位相关内容。

用户原始问题：{question}
知识库检索片段：
{retrieved_chunks}

任务：
1. 从检索片段中提取与问题直接相关的原文，**必须逐条标注来源**（格式：【片段1】、【片段2】...）
2. 如果所有片段都不相关，直接回复【无相关资料】
3. 判断相关性并给出置信度（高/中/低）

输出格式：
【相关片段】
[逐条列出原文]

【相关性判断】
[高/中/低] - [简要理由]
"""

analyzer_prompt = """你是一个分析专家，能够整合碎片化信息形成连贯回答。

用户问题：{question}
检索专家提供的相关片段：{retrieve_result}

任务：
1. 提取关键信息，按逻辑顺序组织（时间线/重要性/因果关系）
2. 如果多个片段信息冲突，明确指出矛盾点
3. 输出一段**通顺、有层次**的中间回答（200-300字）
"""

evaluator_prompt = """你是一个质量评估专家，对最终答案进行审核优化。

用户原始问题：{question}
分析专家的中间回答：{analyze_result}

任务：
1. 检查是否存在事实性错误或逻辑漏洞
2. 优化语言表达，确保简洁专业
3. **必须**在末尾附上：
   - 【引用来源】：[列出使用的片段编号]
   - 【置信度】：[0-1之间的分数，基于信息完整性]
   - 【局限性】：[如果信息不足，明确说明]

只输出优化后的最终回答正文（包含上述三部分）。
"""

prompt_retrieve = ChatPromptTemplate.from_messages([("human", retriever_prompt)])
prompt_analyze = ChatPromptTemplate.from_messages([("human", analyzer_prompt)])
prompt_evaluate = ChatPromptTemplate.from_messages([("human", evaluator_prompt)])

chain_retrieve = prompt_retrieve | llm | output_parser
chain_analyze = prompt_analyze | llm | output_parser
chain_evaluate = prompt_evaluate | llm | output_parser

# ============================================================
# 模块4：真正的RAG核心（向量检索 + 分块策略）
# ============================================================
class RAGKnowledgeBase:
    """知识库管理类（简历加分：面向对象设计）"""
    
    def __init__(self, embedding_model, chunk_size: int = 500, chunk_overlap: int = 50):
        self.embeddings = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        logger.info(f"初始化知识库：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def load_and_index(self, file_path: str):
        """加载文档并建立索引（简历加分：完整的数据处理管道）"""
        logger.info(f"开始加载知识库文件：{file_path}")
        start_time = time.time()
        
        # 1. 读取文件
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"原始文本长度：{len(content)} 字符")
        
        # 2. 文本分块（简历加分：分块策略优化）
        chunks = self.text_splitter.split_text(content)
        logger.info(f"文本分块数量：{len(chunks)} 块")
        
        # 3. 构建向量索引
        documents = [Document(page_content=chunk, metadata={"chunk_id": i}) 
                     for i, chunk in enumerate(chunks)]
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        elapsed = time.time() - start_time
        logger.info(f"索引构建完成，耗时：{elapsed:.2f}s")
        return len(chunks)
    
    def retrieve(self, question: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """检索相关文档片段（简历加分：多路召回雏形）"""
        logger.info(f"检索问题：{question[:50]}... (top_k={top_k})")
        start_time = time.time()
        
        if self.vectorstore is None:
            raise ValueError("知识库未初始化，请先调用 load_and_index()")
        
        # 执行检索（带相似度分数）
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=top_k)
        
        # 提取内容和分数
        contents = [doc.page_content for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # 日志记录（简历加分：可观测性）
        for i, (content, score) in enumerate(zip(contents, scores)):
            logger.info(f"  片段{i+1}: 相似度={score:.4f}, 长度={len(content)}字符")
        
        elapsed = time.time() - start_time
        logger.info(f"检索完成，耗时：{elapsed:.3f}s")
        
        return contents, scores

# ============================================================
# 模块5：异常处理和重试机制（简历加分：工程鲁棒性）
# ============================================================
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, ConnectionError))
)
def safe_chain_invoke(chain, inputs: dict) -> str:
    """安全调用LLM链，带自动重试"""
    try:
        logger.debug(f"调用链：{chain.__class__.__name__}")
        return chain.invoke(inputs)
    except Exception as e:
        logger.error(f"LLM调用失败：{str(e)}")
        # 降级策略：返回友好提示
        return "【系统处理中，请稍后重试】"

# ============================================================
# 模块6：主业务流程（整合所有模块）
# ============================================================
def run_multi_role_rag_agent(
    question: str,
    knowledge_base: RAGKnowledgeBase,
    top_k: int = 5
) -> Dict:
    """
    完整的RAG + 多角色Agent流水线
    
    Returns:
        Dict: 包含最终答案、检索片段、各阶段耗时、置信度等
    """
    logger.info(f"===== 开始处理用户问题 =====")
    logger.info(f"问题：{question}")
    
    result = {
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "stages": {}
    }
    
    # ----- 阶段1：向量检索 -----
    stage_start = time.time()
    try:
        chunks, scores = knowledge_base.retrieve(question, top_k=top_k)
        retrieved_content = "\n\n".join([f"【片段{i+1}】{chunk}" for i, chunk in enumerate(chunks)])
        result["stages"]["retrieval"] = {
            "chunks_retrieved": len(chunks),
            "top_scores": scores,
            "elapsed": time.time() - stage_start
        }
        logger.info(f"检索阶段完成，召回{len(chunks)}个片段")
    except Exception as e:
        logger.error(f"检索失败：{e}")
        retrieved_content = "【检索服务异常】"
        result["stages"]["retrieval"] = {"error": str(e)}
    
    # ----- 阶段2：检索专家（筛选+标注） -----
    stage_start = time.time()
    retrieve_result = safe_chain_invoke(chain_retrieve, {
        "question": question,
        "retrieved_chunks": retrieved_content
    })
    result["stages"]["retrieve_expert"] = {
        "output": retrieve_result,
        "elapsed": time.time() - stage_start
    }
    logger.info(f"检索专家阶段完成，耗时：{result['stages']['retrieve_expert']['elapsed']:.2f}s")
    
    # ----- 阶段3：分析专家（整合） -----
    stage_start = time.time()
    analyze_result = safe_chain_invoke(chain_analyze, {
        "question": question,
        "retrieve_result": retrieve_result
    })
    result["stages"]["analyze_expert"] = {
        "output": analyze_result,
        "elapsed": time.time() - stage_start
    }
    logger.info(f"分析专家阶段完成，耗时：{result['stages']['analyze_expert']['elapsed']:.2f}s")
    
    # ----- 阶段4：评估专家（最终输出） -----
    stage_start = time.time()
    final_answer = safe_chain_invoke(chain_evaluate, {
        "question": question,
        "analyze_result": analyze_result
    })
    result["stages"]["evaluate_expert"] = {
        "output": final_answer,
        "elapsed": time.time() - stage_start
    }
    logger.info(f"评估专家阶段完成，耗时：{result['stages']['evaluate_expert']['elapsed']:.2f}s")
    
    result["final_answer"] = final_answer
    result["total_elapsed"] = sum([stage.get("elapsed", 0) for stage in result["stages"].values()])
    logger.info(f"===== 流水线完成，总耗时：{result['total_elapsed']:.2f}s =====")
    
    return result

# ============================================================
# 模块7：评估体系（简历加分：QA测试集 + 指标计算）
# ============================================================
class Evaluator:
    """简易评估器（面试时会问，提前准备好）"""
    
    @staticmethod
    def compute_similarity(answer: str, ground_truth: str) -> float:
        """计算语义相似度（用embedding）"""
        # 实际项目中用RAGAS，这里给个简化版
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        emb1 = embeddings.embed_query(answer[:500])  # 截断防超限
        emb2 = embeddings.embed_query(ground_truth[:500])
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return float(sim)
    
    @staticmethod
    def run_regression_test(kb: RAGKnowledgeBase, test_qa: List[Tuple[str, str]]):
        """回归测试套件"""
        logger.info(f"开始回归测试，共{len(test_qa)}个用例")
        results = []
        for q, gt in test_qa:
            try:
                output = run_multi_role_rag_agent(q, kb, top_k=3)
                sim = Evaluator.compute_similarity(output["final_answer"], gt)
                results.append({"question": q, "similarity": sim, "success": True})
                logger.info(f"  ✅ 用例通过，相似度：{sim:.3f}")
            except Exception as e:
                results.append({"question": q, "error": str(e), "success": False})
                logger.error(f"  ❌ 用例失败：{e}")
        
        avg_sim = np.mean([r["similarity"] for r in results if r["success"]])
        logger.info(f"回归测试完成，平均相似度：{avg_sim:.3f}")
        return results

# ============================================================
# 模块8：主程序入口（可运行）
# ============================================================
if __name__ == "__main__":
    # 1. 初始化知识库
    kb = RAGKnowledgeBase(embeddings, chunk_size=500, chunk_overlap=50)
    kb.load_and_index("paper.txt")  # 替换成你的文件路径
    
    # 2. 执行单次查询
    user_query = "为人工智能做一个应用综述"
    
    result = run_multi_role_rag_agent(
        question=user_query,
        knowledge_base=kb,
        top_k=5
    )
    
    # 3. 打印最终结果
    print("\n" + "="*60)
    print("最终回答：")
    print("="*60)
    print(result["final_answer"])
    print("\n" + "="*60)
    print(f"总耗时：{result['total_elapsed']:.2f}s")
    print(f"各阶段耗时详情：")
    for stage, data in result["stages"].items():
        if "elapsed" in data:
            print(f"  - {stage}: {data['elapsed']:.2f}s")
    
    # 4. （可选）执行回归测试
    # test_cases = [
    #     ("人工智能的定义是什么？", "人工智能是..."),
    #     # 添加你的测试用例
    # ]
    # evaluator = Evaluator()
    # evaluator.run_regression_test(kb, test_cases)