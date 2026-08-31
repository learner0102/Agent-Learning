# eval.py - RAG 检索质量评测
"""
评测脚本：验证 RAG 检索质量，可选端到端回答正确率。

用法（和 main.py 一样，在 E:\\山理工\\LLM_study\\Codes 目录下运行）：
    python study_line/eni/eval.py                 # 仅评测检索命中率（不调 LLM，快）
    python study_line/eni/eval.py --with-answer   # 额外跑 Agent + LLM 判定回答正确率
    python study_line/eni/eval.py --top-k 5       # 调整检索返回条数
    python study_line/eni/eval.py --limit 5       # 只评测前 5 条，快速验证
    python study_line/eni/eval.py --paper xxx.txt # 指定知识库文档

指标说明：
    Hit@k      = 至少命中 1 个期望关键词的用例占比（检索是否找对文档）
    Recall@k   = 期望关键词的平均召回比例（检索覆盖是否完整）
    回答正确率  = LLM-as-judge 判定 Agent 最终回答与参考答案一致的比例
"""
import argparse
import json
from typing import Dict, List, Tuple

from langchain_openai import ChatOpenAI

from config import logger, API_KEY, BASE_URL, MODEL_NAME
from knowledge_base import KnowledgeBase

PAPER_PATH = r"study_line\eni\paper.txt"


def load_golden(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================== 检索命中率评测 ==================
def eval_retrieval(kb: KnowledgeBase, cases: List[Dict], top_k: int = 3) -> Dict:
    print("\n" + "=" * 70)
    print(f"RAG 检索质量评测  top_k={top_k}  用例数={len(cases)}")
    print("=" * 70)

    hit_cnt = 0
    recall_sum = 0.0
    miss_cases: List[Dict] = []

    for i, case in enumerate(cases, 1):
        q = case["question"]
        expected = [kw.lower() for kw in case["expected_keywords"]]

        contents, scores, _ = kb.retrieve_with_meta(q, top_k=top_k)
        joined = "\n".join(contents).lower()

        hit_kws = [kw for kw in expected if kw in joined]
        hit = len(hit_kws) > 0
        recall = len(hit_kws) / len(expected)

        hit_cnt += int(hit)
        recall_sum += recall

        mark = "✅" if hit else "❌"
        print(f"\n[{i:>2}/{len(cases)}] {mark} {q}")
        print(f"    命中关键词: {hit_kws if hit_kws else '无'}")
        if not hit:
            print(f"    期望关键词: {case['expected_keywords']}")
            miss_cases.append(case)

    n = len(cases)
    hit_rate = hit_cnt / n
    avg_recall = recall_sum / n
    print("\n" + "=" * 70)
    print(f"Hit@{top_k}（至少命中1个关键词） : {hit_cnt}/{n} = {hit_rate:.1%}")
    print(f"Recall@{top_k}（关键词平均召回率）: {avg_recall:.1%}")
    print("=" * 70)

    if miss_cases:
        print(f"\n⚠️ 未命中的 {len(miss_cases)} 个用例（据此改进分块/检索）：")
        for c in miss_cases:
            print(f"  - {c['id']}: {c['question']}")

    return {"hit_rate": hit_rate, "recall": avg_recall}


# ================== 回答正确率评测（LLM-as-judge） ==================
JUDGE_PROMPT = """你是严格的 AI 评测员。判断下面的"模型回答"是否准确回答了"问题"，并且与"参考答案"中的关键事实一致。
允许措辞不同，但关键事实不能缺失或冲突。不要因为风格差异扣分。
只输出一行，格式：数字 空格 一句话理由。数字 1 表示正确，0 表示错误。

问题：{question}
参考答案：{reference}
模型回答：{answer}"""


def create_judge() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=0.0,  # 判定需要确定性
        max_tokens=200,
        timeout=60.0,
    )


def judge_one(judge: ChatOpenAI, question: str, reference: str, answer: str) -> Tuple[int, str]:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    try:
        resp = judge.invoke(prompt)
        text = (resp.content or "").strip()
        score = 1 if text.startswith("1") else 0
        reason = text[2:].strip() if len(text) > 2 else ""
        return score, reason
    except Exception as e:
        logger.error(f"判定失败: {e}")
        return 0, f"判定异常: {e}"


def eval_answer(agent, cases: List[Dict], judge: ChatOpenAI) -> Dict:
    print("\n" + "=" * 70)
    print("端到端回答正确率（LLM-as-judge）")
    print("=" * 70)

    correct = 0
    n = len(cases)
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{n}] {case['question']}")
        result = agent.run(case["question"])
        answer = result["final_answer"]
        score, reason = judge_one(judge, case["question"], case["reference_answer"], answer)
        correct += score
        print(f"    判定: {score}  {reason}")
        print(f"    回答前80字: {answer[:80]}...")

    acc = correct / n if n else 0
    print("\n" + "=" * 70)
    print(f"回答正确率: {correct}/{n} = {acc:.1%}")
    print("=" * 70)
    return {"answer_accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--golden", default=r"study_line\eni\golden.json", help="评测集路径")
    parser.add_argument("--paper", default=PAPER_PATH, help="知识库文档路径")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回条数")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条（0=全部）")
    parser.add_argument("--with-answer",default=False, action="store_true", help="额外运行 Agent 评测回答正确率")
    args = parser.parse_args()

    cases = load_golden(args.golden)
    if args.limit > 0:
        cases = cases[:args.limit]

    logger.info(f"加载评测集: {len(cases)} 条用例")

    # 初始化知识库（与 main.py 相同的参数）
    kb = KnowledgeBase(chunk_size=500, chunk_overlap=50)
    kb.load_and_index(args.paper)

    # 1) 检索命中率
    ret_metrics = eval_retrieval(kb, cases, top_k=args.top_k)

    # 2) 端到端回答正确率
    if args.with_answer:
        from tools import init_tools
        from agent import RAGAgent

        init_tools(kb)
        agent = RAGAgent(max_steps=5)
        judge = create_judge()
        ans_metrics = eval_answer(agent, cases, judge)
        print(f"\n📊 汇总: Hit@{args.top_k}={ret_metrics['hit_rate']:.1%} | "
              f"Recall@{args.top_k}={ret_metrics['recall']:.1%} | "
              f"回答正确率={ans_metrics['answer_accuracy']:.1%}")
    else:
        print(f"\n📊 汇总: Hit@{args.top_k}={ret_metrics['hit_rate']:.1%} | "
              f"Recall@{args.top_k}={ret_metrics['recall']:.1%}")
        print("💡 想看回答正确率，加 --with-answer 参数（会调用模型，需要 API）")


if __name__ == "__main__":
    main()
