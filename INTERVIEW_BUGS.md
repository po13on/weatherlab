# 气象实验复核 · 面试缺陷日记

每条五栏：现象、错误判断、根因、修复、面试官下一问。先记事故，再改代码。数字和结论以仓库为准，不现场编。

完整条目的结构化版本见页面「缺陷日记」或 `interview_bugs.py`。

## B1 dotenv 不覆盖已有 Key

- 现象：401，报错 key 后缀不是 `.env` 里那串。
- 错误判断：用户 Key 无效。
- 根因：`load_dotenv` 默认不覆盖进程里已有的 `OPENAI_API_KEY`。
- 修复：`override=True`。
- 下一问：CI 与本机同名变量以谁为准？

## B2 官方模型 id 不是产品营销名

- 现象：`deepseek-v4.1-flash` 不能当 ChatCompletions 模型名。
- 错误判断：SDK 太旧。
- 根因：官方 id 是 `deepseek-flash`。
- 修复：别名映射；health 同时返回 requested / resolved。
- 下一问：计费和路由按哪个 id？

## B3 默认 thinking 干扰工具循环

- 现象：Tool Calling 不稳。
- 错误判断：prompt 不够凶。
- 根因：Flash 默认 thinking。
- 修复：`thinking.disabled`。
- 下一问：何时开 thinking？

## B4 Q7 被无关手册「救活」

- 现象：无日志 run 仍 `supported`。
- 错误判断：Critic 没跑。
- 根因：手册命中了别的故障，指针存在就算过。
- 修复：本 run 日志/清单都缺则硬拒答。
- 下一问：手册能否单独支持根因？

## B5 真指针 + 假根因

- 现象：引用 OOM 日志，结论写成硬件故障。
- 错误判断：必须再加一个裁判模型。
- 根因：Critic 只校验 id。
- 修复：矛盾文本拒绝。
- 下一问：规则误杀怎么收边？

## B6 注入句放错位置

- 现象：口述是日志注入，数据在用户问题里。
- 错误判断：模型会自己生成注入句。
- 根因：fixture 与手册不一致。
- 修复：写入 `run-fail-oom` 日志。
- 下一问：直接 vs 间接注入的控制点？

## B7 innerHTML

- 现象：模型/日志进 HTML。
- 错误判断：本地没有 XSS。
- 根因：字符串拼接 DOM。
- 修复：`textContent`。
- 下一问：SSE JSON 要不要再转义？

## B8 同步分析堵请求

- 现象：页面一直转圈。
- 错误判断：换更快模型。
- 根因：分析在 HTTP 线程跑完。
- 修复：`job_id` + Worker + SSE。
- 下一问：崩溃后恢复什么、不恢复什么？
