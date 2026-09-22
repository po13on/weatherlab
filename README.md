# 气象实验复核

天气模型实验 / 运行分析 Agent。可运行完整 Demo，不是线上系统。

- 手写 ReAct + 真实 Tool Calling + 确定性工具 + Citation Critic（主路径）
- FastAPI 异步入口：`POST /analyze` 立即返回 `job_id`，任务在线程池跑，SSE 推轨迹
- 存储可切换：默认 SQLite；`STORAGE=postgres` 后同一套 API 走 PostgreSQL
- 缓存可切换：无 `REDIS_URL` 时用内存实现四用途；有 Redis 则写 `job:` / `ckpt:` / `idem:` / `rl:`
- 手册按页切块后做向量检索。默认哈希向量。扫描页先渲染成图，再用模板 OCR 读回，不把原稿直接当索引文本。Python 3.10+ 可切 LlamaIndex；装了 chromadb 且 `RAG_BACKEND=chroma` 时走 Chroma。实验表不进向量库
- LangGraph 只作为可选评测对照，不替换手写循环
- 20 题 fixture 评测 + 10 条手册 Recall@K 与页级 citation_precision；数字只引用最近一次 `eval_runs`
- 从轨迹导出 SFT jsonl，**未做微调**
- 角色权限、工具超时重试、HITL、缺陷日记

完整口述见 `../求职作战包/WeatherLab_面试设计手册.md`。框架对照见 `FRAMEWORK_NOTES.md`。

## Python 版本

- **核心 Demo / pytest**：当前机器可用 3.8 跑通（不含 LangChain 包）。
- **JD 框架层（LangChain / LangGraph / LlamaIndex / Chroma）**：需要 **Python 3.10+**。用 `scripts/bootstrap_py312.ps1` 或自行安装 3.12 再建 `.venv`。

不要在 3.8 上装过期 LangChain 0.0.x 来「凑关键词」。

## 启动（一期，无 Docker）

```powershell
cd e:\work\weatherlab
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env` 至少填 Key。`STORAGE=sqlite`，`REDIS_URL` 留空。`RAG_BACKEND=hash`（默认）。`llama` 或 `chroma` 只有在对应包已安装且环境变量显式设置时才启用。

```powershell
uvicorn app:app --reload
```

打开 http://127.0.0.1:8000

- 分析页：预置题、历史回放、可选 HITL、guest 角色
- 评测页：`POST /eval/runs` 的 kinds 可为 `system` / `baseline` / `retrieval` / `langgraph`
- 缺陷日记：只读 B1–B8

## 二期（真 PostgreSQL + Redis）

```powershell
docker compose up -d
```

这不是微服务。Compose 只起数据库和 Redis。不要说已经拆了 api/worker 集群。

## HITL

`HITL_MODE=off|manual|eval_skip`

## 测试

```powershell
python -m pytest -q
```

评测打分器测试用 mock agent，不耗额度。有 Key 时再点页面评测。

导出微调数据（不训练）：

```powershell
python scripts/export_sft_jsonl.py --out data/sft.jsonl
```

## 面试怎么讲

可以说：业务助手型 Agent；主循环手写；手册是分页检索，扫描页有 OCR；实验数字仍由确定性工具算。异步是 FastAPI 入口 + 线程池，不是高并发。

不能说：已上线、微服务、微调提升了准确率、没跑过的准确率、PaddleOCR 或视觉大模型已经接入、Chroma/Milvus 已在默认路径、生产用 LangGraph。
