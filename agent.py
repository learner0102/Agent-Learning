# agent.py - 基于 LangGraph + 原生 Function Calling
from typing import Annotated, Dict, Any, List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config import logger, API_KEY, BASE_URL, MODEL_NAME, TEMPERATURE, MAX_AGENT_STEPS
from tools import get_tools


class RAGAgent:
    """
    基于 LangGraph 的 Agent，使用「原生 Function Calling」驱动工具调用。

    对比旧版手写正则解析 ReAct 文本（Action:/Final Answer:）：
      - 工具调用由模型结构化输出 tool_calls，无需解析字符串，解析稳定
      - tool_call_id 由框架生成，ToolMessage 严格匹配，天然支持多工具并行
      - 终止条件直接看最后一个 AIMessage 是否带 tool_calls，逻辑更简单可靠
    """

    def __init__(self, max_steps: int = MAX_AGENT_STEPS):
        # ChatOpenAI 走 OpenAI 兼容协议接入 Qwen（DashScope）
        self.llm = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            timeout=60.0,
            max_tokens=2048,
        )
        self.tools = get_tools()
        # 把工具 schema 绑定到模型：模型输出结构化的 tool_calls
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.max_steps = max_steps
        self.thought_process: List[str] = []
        self.tool_results: List[str] = []
        self.current_step = 0

        self.graph = self._build_graph()
        logger.info(f"Agent初始化完成，工具数: {len(self.tools)}，最大步数: {max_steps}")

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------
    def _build_graph(self):
        class GraphState(TypedDict):
            # add_messages reducer：节点返回 {"messages": [新消息]} 时自动追加到历史
            messages: Annotated[List, add_messages]
            step: int

        def agent_node(state: GraphState) -> GraphState:
            """Agent 决策节点：调用绑定了工具的 LLM"""
            self.current_step += 1
            logger.info(f"Agent 第 {self.current_step}/{self.max_steps} 步决策")

            # invoke 返回 AIMessage：要么带 tool_calls，要么是最终回答 content
            ai_msg = self.llm_with_tools.invoke(state["messages"])

            # 记录思考过程
            if ai_msg.tool_calls:
                self.thought_process.append(
                    "调用工具: " + ", ".join(tc["name"] for tc in ai_msg.tool_calls)
                )
            else:
                self.thought_process.append(ai_msg.content or "")

            return {"messages": [ai_msg], "step": self.current_step}

        def tool_node(state: GraphState) -> GraphState:
            """工具执行节点：遍历 AIMessage 里的所有 tool_calls，逐个执行并回填结果"""
            last_msg = state["messages"][-1]
            tool_messages: List[ToolMessage] = []

            for tc in last_msg.tool_calls:
                tool = next((t for t in self.tools if t.name == tc["name"]), None)
                if tool is None:
                    result = f"未知工具: {tc['name']}"
                    logger.error(result)
                else:
                    logger.info(f"执行工具 {tool.name}({tc['args']})")
                    try:
                        result = tool.invoke(tc["args"])
                    except Exception as e:
                        logger.error(f"工具执行失败: {e}")
                        result = f"工具执行出错: {str(e)}"

                self.tool_results.append(result)
                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

            return {"messages": tool_messages}

        def should_continue(state: GraphState) -> str:
            """决定流程走向：还有工具调用就继续，否则结束"""
            if state["step"] >= self.max_steps:
                logger.warning(f"达到最大步数 {self.max_steps}，强制结束")
                return "end"

            last_msg = state["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                return "tool"
            return "end"

        builder = StateGraph(GraphState)
        builder.add_node("agent", agent_node)
        builder.add_node("tool", tool_node)
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {
            "tool": "tool",
            "end": END,
        })
        builder.add_edge("tool", "agent")

        return builder.compile()

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def run(self, question: str, extra_context: str = "", session_context: str = "") -> Dict[str, Any]:
        logger.info(f"Agent启动，用户问题: {question}")

        # 重置状态
        self.thought_process = []
        self.tool_results = []
        self.current_step = 0

        system_prompt = (
            "你是一个智能问答Agent。当问题涉及知识库中的具体内容时，"
            "请调用 search_knowledge_base 工具检索后再回答；"
            "检索结果足以回答时，直接给出最终回答。"
        )

        user_prompt = f"""用户问题：{question}

会话历史（短期记忆）：
{session_context}

检索出来的额外信息（长期记忆）：
{extra_context}

请基于会话历史和检索到的信息回答。若需要知识库中的具体内容，请调用工具(如果你调用了工具，需要明确告诉我工具名称)。"""

        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            result = self.graph.invoke({
                "messages": initial_messages,
                "step": 0,
            })

            final_answer = self._extract_final_answer(result["messages"])
            return {
                "question": question,
                "final_answer": final_answer,
                "steps": self.current_step,
                "thought_process": self.thought_process,
                "tool_results": self.tool_results,
            }
        except Exception as e:
            logger.error(f"Agent运行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "question": question,
                "final_answer": f"运行出错: {str(e)}",
                "steps": 0,
                "thought_process": [],
                "tool_results": [],
            }

    def _extract_final_answer(self, messages: List) -> str:
        """从消息历史里提取最终回答：倒序找第一个带文本内容的 AIMessage"""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "抱歉，未能生成有效回答"
