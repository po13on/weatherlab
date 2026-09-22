# 手写 ReAct 与 LangGraph / LlamaIndex 对照

气象实验复核的**系统路径仍是手写循环**（`agent.py`）。框架只做两件事：手册检索，以及可选的 `kind=langgraph` 评测对照。不要口述成「生产用 LangGraph 编排」。

## 循环怎么对应

手写 `while remaining_steps` 对应 LangGraph 的 `StateGraph`：

| 手写 | LangGraph 节点 / 边 |
|---|---|
| `BUILD_CONTEXT` 拼 messages | 初始 `state["messages"]` |
| `PLAN`：`completer(messages, tools)` | `model` 节点 |
| `EXECUTE_STEP`：`dispatch(...)` | `tools` 节点 |
| 重复工具 / 步数用尽 | 条件边到 `END` |
| Citation Critic | `tools` 节点里的 `critique(...)`，不是第二个 Agent |
| `_finish` + 证据门 | graph.invoke 之后同样调用 `_finish` |

源码入口：

- 手写：`agent.run_agent`
- 对照：`runners.langgraph_baseline.build_langgraph`（需 Python 3.10+ 且已安装 langgraph）

LangChain 1.x 的 `create_agent` 底层也是「模型节点 → 工具节点 → 循环直到没有 tool_calls」。本仓库把同一结构写进 StateGraph，工具实现仍是 `tools.dispatch`，权限仍走 `policy.py`。

## 手册检索

`retrieve_runbook` 走 `rag.search_runbooks`，单位是页。手册正文在 `求职作战包/weatherlab_fixtures/runbooks/*.md`。`source: scan` 的页先画成 PNG，再由 `rag.ocr` 的模板匹配读回来，索引里只有 OCR 文本。`ocr_page` 会按同一张图重读。默认哈希向量。Python 3.10+ 且安装了 `llama-index-core` 时，`RAG_BACKEND=llama` 可用 LlamaIndex。安装了 `chromadb` 时，`RAG_BACKEND=chroma` 会把页块写入内存 Chroma。实验表 **不** 进向量库。

召回只报 `POST /eval/runs` 里 `kind=retrieval` 的最近一次 `recall_at_k` / `hit_rate` / `citation_precision`。`citation_precision` 是 top-1 的文档和页是否同时命中金标。默认排序是查询词是否出现在该页，哈希余弦只用来打破平局。

## 模型 API

`llm.py` 的 `MODEL_ALIASES` 把 DeepSeek / 可选 Qwen / OpenAI 兼容名映射到实际 `model` 字段。DeepSeek 必须关 thinking，否则 function call 不稳定。没有在本仓库训练 Llama 3。

## 微调

`python scripts/export_sft_jsonl.py` 从 job 轨迹导出 OpenAI chat jsonl。文件里带 `not_trained: true`。没有 LoRA、没有报涨点。

## 异步与「微服务」

FastAPI 入口是 `async def`，分析任务用线程池执行同步 Agent，避免堵住事件循环。Compose 里仍是 PostgreSQL + Redis 可选依赖，**不是**拆好的微服务。面试说「可拆成 api/worker 两进程的 Demo」。
