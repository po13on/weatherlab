# 交接说明

这份说明给下一台电脑、下一个模型用。仓库里的可运行项目叫**气象实验复核**。用户可见名称不要改回 WeatherLab。代码目录和 GitHub 仓库名保持 `weatherlab`。

## 当前事实

- GitHub：https://github.com/po13on/weatherlab
- 账号：`po13on`。提交身份用本机已有的 Git 用户名，不要改 `git config`。
- 原开发机路径：`e:\work\weatherlab`。另一台电脑克隆后路径可以不同。
- 原机还有一份求职材料在 `e:\work\求职作战包`。那份材料**没有**放进这个仓库，里面有未填完的姓名、电话、实习和奖项年份。不要把那些占位符编成事实，也不要把它推到 GitHub。
- 实验数据的正本现在是仓库里的 `fixtures/`。若设置 `WEATHERLAB_FIXTURES` 且该目录里有 `experiments.json`，启动进程时优先用它。这个路径在 import 时定下来，改环境变量后要重启进程。只有仓库里没有 `fixtures/experiments.json` 时，才回退到原机的 `求职作战包/weatherlab_fixtures`。

## 另一台电脑怎么跑起来

需要 Python。核心 Demo 和 pytest 在 **Python 3.8** 上跑过。LangChain、LangGraph、LlamaIndex、Chroma 这些可选依赖要求 **Python 3.10+**，原机用的是 3.12。不要在 3.8 上装 LangChain 0.0.x 来凑关键词。

```powershell
git clone https://github.com/po13on/weatherlab.git
cd weatherlab
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m pytest -q
uvicorn app:app --reload
```

浏览器打开 http://127.0.0.1:8000 。`.env` 里填模型 Key。没有 Key 时，页面和大部分测试仍可跑，真正调用模型的分析会停住。`.env` 已在 `.gitignore` 里，不要提交。

`.env.example` 的默认值：

- `OPENAI_BASE_URL=https://api.deepseek.com`
- `OPENAI_MODEL=deepseek-v4.1-flash`
- `HITL_MODE=eval_skip`
- `STORAGE=sqlite`
- `REDIS_URL` 留空，此时用内存实现状态、checkpoint、幂等、限流

PostgreSQL 和 Redis 只有执行 `docker compose up -d` 并改环境变量之后才是真服务。Compose 只起这两个依赖，不是微服务拆分。

## 代码地图

| 路径 | 作用 |
| --- | --- |
| `app.py` | FastAPI。`POST /analyze` 返回 `job_id`，线程池里跑分析，SSE 推轨迹 |
| `agent.py` | 手写 ReAct。模型只决定下一步调哪个工具 |
| `tools.py` | 只读工具：配置差、指标对比、缺测、日志首错，以及 `ocr_page` |
| `policy.py` | 角色。guest 不能读日志和扫描页 |
| `critic.py` | Citation Critic。结论必须带本轮 `evidence_id`，并核对手册的文档和页码 |
| `jobs.py` / `db.py` / `cache.py` | 异步任务、SQLite 或 PostgreSQL、Redis 或内存 |
| `llm.py` | OpenAI 兼容接口，原机指向 DeepSeek |
| `rag/ingest.py` | 手册按页切块。`source:scan` 的页先渲染成图，再用 OCR 文本入库 |
| `rag/ocr.py` | 离线模板匹配 OCR，引擎名 `template`。不是 PaddleOCR，也不是视觉大模型 |
| `rag/runbook_index.py` | 默认 `RAG_BACKEND=hash`：词面覆盖加很小的哈希余弦。`llama` / `chroma` 只有显式设置且包装上才走 |
| `prompts/` | `wl-sys-v3`、`wl-critic-v2` |
| `eval_suite.py` / `eval_cases.py` / `eval_retrieval.py` | 20 道合成题，外加 10 条手册检索题 |
| `runners/langgraph_baseline.py` | 可选对照，不替换主循环。3.8 上相关测试会跳过 |
| `scripts/export_sft_jsonl.py` | 从轨迹导出 jsonl。导出不等于训练 |
| `static/index.html` | 本地演示页 |
| `fixtures/` | `experiments.json` 和五本手册。扫描页金句在 `runbooks/rb-oom.md` 第 3 页：`CUDA OOM HALVE BATCH` |
| `FRAMEWORK_NOTES.md` | 和 LangGraph / LlamaIndex 的源码对照，给面试用 |
| `INTERVIEW_BUGS.md` | 缺陷日记 |

手册检索的向量只嵌入引用句，不把共享标题嵌进去。否则标题会把别的页排到金页前面。

## 上次测试

提交前在原机 Python 3.8 上跑过 `python -m pytest -q`：**58 passed，1 skipped**。跳过的是 LangGraph（3.8 没有那个包）。换电脑后先自己再跑一次，不要沿用这句当作新机器的结果。

评测数字只引用最近一次 `eval_runs`。仓库里没有一份可以写进简历的准确率。设计过的比例、没落盘的 Recall，都不要写进简历或 README。

## 可以说的和不能说的

可以说：这是业务助手型 Agent；主循环手写；手册按页检索；扫描页有模板 OCR；实验数字由确定性工具计算；异步是 FastAPI 加线程池。

不能说：已经上线或进了生产；拆成了微服务；微调提高了准确率；没跑过的准确率或召回率；默认在用 Chroma、Milvus、LlamaIndex；主路径是 LangGraph、CrewAI 或 Kafka；OCR 已经换成 PaddleOCR 或视觉大模型。

飞书实习只说明「多实验对照、缺测、日志里第一条错误」这类问题从哪来。不要把实习单位的内部系统、集群或真实数据写进仓库和简历。

## 给下一个模型的工作边界

- 先读这份说明、`README.md` 和 `FRAMEWORK_NOTES.md`，再改代码。
- 改检索或 OCR 后，跑 `python -m pytest -q`。10 条手册题的 `citation_precision` 必须是 1.0。查询 `CUDA OOM correction head halve batch_size` 的第一条命中必须是 `rb-oom` 第 2 页。扫描页上的 `CUDA OOM HALVE BATCH` 在第 3 页，那是 OCR 重读的对象，不要把它改成检索金页。
- 不要为了岗位关键词把主循环换掉。LangGraph 和 LlamaIndex 保持可选对照。
- 不要提交 `.env`、`data/`、`.venv/`、`.chroma/`、`__pycache__/`。
- 简历如果要补 GitHub 链接，用上面的仓库地址。姓名、电话、本科、实习单位、奖项年份仍然空着，没有新事实就不要填。
