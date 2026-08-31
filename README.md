# RAG Agent 智能问答系统（LangGraph + 原生 Function Calling）

基于 **LangGraph + LangChain** 的 RAG 智能问答 Agent，集成多步 Agent 决策、知识库检索增强（RAG）与分层记忆系统。模型接入通义千问（Qwen），Embedding 使用本地 Qwen3-Embedding-0.6B，全程本地化部署（CUDA）。

## ✨ 功能特性

- 💬 交互式命令行问答，支持多轮会话
- 🔍 基于知识库的 RAG 检索增强生成，返回相关度分数
- 🤖 LangGraph 多步 Agent 决策循环，使用**原生 Function Calling** 驱动工具调用
- 🧠 分层记忆系统：
  - `working` 短期会话记忆（纯内存，重启清空）
  - `episodic / semantic` 长期记忆（重要度 ≥ 0.7 自动持久化到本地 JSON）
  - 检索时**语义召回优先、关键词兜底**，实现跨会话"记住用户说过什么"
- 📄 中文友好的智能分块（Markdown 标题感知 + CJK Token 估算）
- 🛡️ 批量向量化构建 FAISS 索引，单批失败自动跳过，不中断整体流程

## 🏗️ 架构流程

```
用户输入
   │
   ▼
┌─────────────────────────────┐
│ 记忆层  MemoryManager       │
│  ├─ 写入短期记忆 working     │
│  ├─ 检索长期记忆             │
│  │   语义召回(FAISS) → 兜底  │
│  │   (关键词 bigram)         │
│  └─ 产出 session_context +   │
│      extra_context           │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Agent 层  RAGAgent          │
│ LangGraph StateGraph        │
│  agent_node:                │
│   ChatOpenAI + bind_tools   │
│      │                      │
│      ├─ 有 tool_calls ─►    │
│      │    tool_node         │
│      │    执行工具           │
│      │    ToolMessage 回填  │
│      │    └──► 回到 agent   │
│      │                      │
│      └─ 生成最终回答 ─► 输出 │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 知识库层  KnowledgeBase     │
│  ├─ 分块                    │
│  │   字符递归 / 标题感知     │
│  ├─ Embedding               │
│  │   本地 Qwen3-Embedding   │
│  └─ FAISS 向量检索 top_k    │
└─────────────────────────────┘
```

## 📂 目录结构

```
eni/
├── main.py             # 程序入口：初始化 + 交互式问答循环
├── agent.py            # RAGAgent：LangGraph 状态机 + 原生 Function Calling
├── tools.py            # 工具定义：search_knowledge_base（闭包注入知识库）
├── knowledge_base.py   # 知识库：分块 / Embedding / FAISS 索引 / 检索
├── chunks.py           # 自研分块：Markdown 标题感知 + CJK Token 估算
├── memory_manager.py   # 分层记忆：短期/长期、重要度持久化、语义召回
├── config.py           # 环境变量 + 日志配置
├── paper.txt           # 示例知识库文档
├── memory_store/       # 长期记忆落盘目录（mem_<user_id>.json）
└── logs/               # 运行日志
```

## 🛠️ 技术栈

| 分类 | 技术 |
| --- | --- |
| 语言 / 框架 | Python 3.10+, LangChain, LangGraph |
| Agent | LangGraph StateGraph、条件边、原生 Function Calling（`bind_tools`）、最大步数限流 |
| RAG | FAISS 向量检索、HuggingFace Embeddings（本地 Qwen3-Embedding-0.6B / SentenceTransformers） |
| 分块 | RecursiveCharacterTextSplitter + 自研 Markdown 标题感知 Token 分块、CJK Token 估算 |
| 记忆 | 分层记忆（working / episodic / semantic）、重要度阈值持久化、语义召回 + 关键词兜底 |
| 模型接入 | 通义千问 Qwen（OpenAI 兼容协议 / DashScope），langchain-openai `ChatOpenAI` |
| 工程 | 批量向量化 + 单批容错、logging、dotenv 配置分离、CUDA 本地推理 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install langchain langgraph langchain-openai langchain-community \
            langchain-text-splitters faiss-cpu sentence-transformers \
            openai python-dotenv
```

### 2. 配置环境变量

在 `.env` 中配置：

```ini
Qwen_API_KEY=你的DashScope_API_KEY
DB_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=你的模型名
```

### 3. 运行

```bash
python main.py
```

### 4. 使用命令

| 命令 | 说明 |
| --- | --- |
| `exit` / `quit` | 退出系统 |
| `#showsession` | 查看当前会话短期记忆 |
| `#save <问题>` | 将本轮问答保存为长期记忆（示例：`#save 我叫小明`） |

直接输入问题即触发问答；命中"我叫 / 我是 / 我喜欢"等个人关键词时，自动存入长期记忆。

## 🧠 记忆系统设计

采用**分层记忆**，各司其职：

| 类型 | 存储位置 | 生命周期 | 用途 |
| --- | --- | --- | --- |
| `working`（短期会话） | 内存 | 当前会话，重启清空 | 保存完整对话，用于多轮上下文 |
| `episodic`（情景） | 内存 + 磁盘 | 重要度 ≥ 0.7 落盘 | 重要问答、文档加载事件 |
| `semantic`（语义） | 内存 + 磁盘 | 重要度 ≥ 0.7 落盘 | 用户个人事实、知识点 |

检索策略：**先语义召回**（用与知识库共享的本地 Embedding 构建 FAISS 索引，对长期记忆做相似度检索），**再关键词兜底**（中英文分词 + bigram 匹配），避免"换个说法就失忆"。

## 💡 设计要点与权衡

- **为什么用原生 Function Calling 而不是手写正则 ReAct**：工具调用由模型结构化输出 `tool_calls`，框架用 `tool_call_id` 严格匹配回填，解析稳定、天然支持多工具并行，是生产级 Agent 的标准做法。
- **为什么用本地 Embedding**：数据不出本地、无 API 调用成本、可离线；代价是需要显卡资源。与记忆系统共用同一模型实例，避免重复加载。
- **为什么自研分块**：固定字符切分会切断语义，尤其是中文没有天然空格。自研方案感知 Markdown 标题层次，按 CJK 字符近似 Token 数分块，保留标题路径做溯源。
- **为什么 FAISS**：轻量、内存态、社区成熟，适合单机百万级以内向量；相比 Elasticsearch 免去服务运维成本。

## 🗺️ Roadmap（规划中）

- [ ] 检索质量评测（golden set / RAGAS），量化检索命中率与回答质量
- [ ] 混合检索（向量 + BM25）与 Rerank 精排
- [ ] LLM 驱动的记忆提取 / 合并 / 去重（参考 Mem0 / MemGPT）
- [ ] LangGraph Checkpointer 实现会话级持久化
- [ ] FastAPI 服务化 + 流式输出（streaming）
- [ ] 可观测性接入（Langfuse / LangSmith）
- [ ] 单元测试与 CI
