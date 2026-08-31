import json
from datetime import datetime
from config import logger
from knowledge_base import KnowledgeBase
from agent import RAGAgent
from tools import init_tools
from memory_manager import MemoryManager  # 导入记忆管理器


def main():
    """主程序入口"""
    print("=" * 60)
    print("Agent + RAG 智能问答系统（自动读取长期记忆）")
    print("=" * 60)

    # 1. 初始化知识库
    logger.info("【步骤1】加载知识库")
    kb = KnowledgeBase(chunk_size=500, chunk_overlap=50)
    kb.load_and_index(r"study_line\eni\paper.txt")

    # 记忆初始化（复用知识库的 embedding 模型，避免把大模型加载两遍）
    user_id = "local_user_01"
    session_id = f"ses_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mem = MemoryManager(user_id=user_id, embeddings=kb.embeddings)
    logger.info(f"【记忆初始化】user_id={user_id}, session_id={session_id}")
    print("💡指令提示：#save 将本轮问答保存为本地长期记忆；#showsession 查看会话对话；exit退出\n")

    # 加载文档这个事件属于重要信息，存入长期记忆（importance=0.9 ≥阈值，自动落盘json）
    mem.add(
        content=f"已加载文档：study_line\\eni\\paper.txt",
        memory_type="episodic",
        session_id=session_id,
        importance=0.9,
        event_type="document_loaded"
    )

    # 2. 初始化工具（注入知识库）
    logger.info("【步骤2】初始化Agent工具")
    init_tools(kb)

    # 3. 初始化Agent
    logger.info("【步骤3】初始化Agent")
    agent = RAGAgent(max_steps=5)

    # 4. 交互式问答
    print("\n" + "=" * 60)
    print("Agent已就绪，输入问题开始问答（输入 exit 退出）")
    print("=" * 60)

    while True:
        question = input("\n💬 用户: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            print("退出系统，当前会话短期记忆全部丢弃；已保存的长期记忆保留在本地文件。")
            break

        if not question:
            continue

        # 内置命令处理
        if question.startswith("#showsession"):
            print("\n📒 当前会话短期记忆（内存，重启丢失）：")
            for item in mem.get_short_term():
                print(f"- {item['content']}")
            continue

        # 内置命令：#save —— 先剥离前缀，得到真正要问的问题
        is_save = question.startswith("#save")
        real_question = question.removeprefix("#save").strip() if is_save else question
        if is_save and not real_question:
            print("⚠️ #save 后面要跟具体问题，例如：#save 数字孪生是什么")
            continue

        print("\n🤖 Agent思考中...")

        # --------------------------
        # 1. 用户提问写入【短期会话记忆 working】，仅内存，不写磁盘
        mem.add(
            content=f"用户：{real_question}",
            memory_type="working",
            session_id=session_id,
            importance=0.5
        )

        # 2. ✅自动检索长期记忆（磁盘持久化的重要记忆），不需要用户手动#recall
        session_context = mem.get_session_context()
        recall_result = mem.search(real_question, limit=3)
        extra_context = f"\n【历史重要记忆】\n{recall_result}\n"

        # 3. 将检索出来的长期记忆作为额外上下文传给Agent
        result = agent.run(real_question, extra_context=extra_context, session_context=session_context)

        # 4. Agent回答写入【短期会话记忆 working】，仅内存
        mem.add(
            content=f"Agent：{result['final_answer']}",
            memory_type="working",
            session_id=session_id,
            importance=0.5
        )
        # --------------------------

        # 打印结果
        print("\n" + "-" * 40)
        print("📝 最终回答:")
        print("-" * 40)
        print(result["final_answer"])
        print("-" * 40)
        print(f"⏱️ 步骤数: {result['steps']}")
        print(f"🛠️ 工具调用次数: {len(result['tool_results'])}")

        # 展示思考过程
        if result["thought_process"]:
            print("\n🧠 思考过程:")
            for i, thought in enumerate(result["thought_process"], 1):
                print(f"  Step {i}: {thought[:150]}...")

        # #save：把本轮问答手动保存到本地长期记忆
        if is_save:
            mem.add(
                content=f"重要问答：{real_question} → {result['final_answer'][:400]}",
                memory_type="episodic",
                session_id=session_id,
                importance=0.85,
                event_type="qa_interaction"
            )
            print("✅ 本轮问答已保存到本地长期记忆！")

        # 个人事实自动保存的判断也要用 real_question
        personal_keywords = ["我叫", "我是", "我喜欢", "我的名字", "我住在", "我工作在", "我讨厌", "我不爱", "我姓"]
        if any(kw in real_question for kw in personal_keywords):
            mem.add(
                content=f"用户提到：{real_question}，Agent回答：{result['final_answer'][:400]}",
                memory_type="semantic", session_id=session_id,
                importance=0.8, event_type="user_fact"
            )
            print(" 检测到个人事实，已自动保存到长期记忆。")


if __name__ == "__main__":
    main()
