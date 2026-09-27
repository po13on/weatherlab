# PDF Agent 用法

在 `pdf-agent` 目录里用命令行运行。这个子目录没有 HTTP 服务，也不要把它说成气象项目的 `uvicorn app:app`。

检索默认用哈希嵌入。`eval` 和 `recall-docvqa` 打出来的召回是这一次检索的计数，不是 Qwen 的成绩。

## 安装

需要 Python 3.10 或更高版本。

```bash
cd pdf-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

模型密钥只读环境变量 `OPENAI_API_KEY`。可选 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。没有密钥时，`ask` 会用脚本模型把工具顺序跑完。不要把密钥写进仓库。

## ingest

分页、对图片页做 OCR、切块，再写入 `--out` 目录。`--state` 和 `--out` 都是必填。`--ocr` 只能是 `fake`（默认）或 `rapidocr`。

```bash
.venv/bin/python -m btom_pdf_agent ingest your.pdf --state examples/run_state.json --out runs/demo
```

`fake` 只认本项目画出来的字格图，不下载 OCR 权重。文本页走 PDF 文本层。

## ask

第一个参数是 `ingest` 写出的目录，第二个是问题。加上 `--fake` 时不请求模型。没有 `OPENAI_API_KEY` 时也会走脚本模型。

```bash
.venv/bin/python -m btom_pdf_agent ask runs/demo "请核对第 2 页注意力图和账本" --fake
```

有密钥时去掉 `--fake` 才会请求模型。

## eval

用仓库里的夹具题检查引用，接着跑 Hi-ToM 故事页检索。数据文件不在仓库里时，对应集合会打印跳过，不会另造数据。

```bash
.venv/bin/python -m btom_pdf_agent eval
```

可选参数的默认值是 `--dir examples/eval` 和 `--state examples/run_state.json`。夹具那一行的格式是 `夹具引用命中 N/3`。后面的 Recall 行来自哈希嵌入检索，只对应当次运行。

## audit-consistency

对照 PDF 里的注意力柱状图和该页结论文字。预测本身不读金标。`--gold` 只在预测完成后打开，用来算不一致页的集合召回。

```bash
.venv/bin/python -m btom_pdf_agent audit-consistency chart.pdf
.venv/bin/python -m btom_pdf_agent audit-consistency chart.pdf --gold gold_inconsistent_ids.json
```

## recall-docvqa

这是另一条已经存在的命令，做的是旧 DocVQA 文本子集上的哈希嵌入检索 Recall。它不看图，不算 ANLS，也不是下面的 Qwen 准确率。

```bash
.venv/bin/python -m btom_pdf_agent recall-docvqa --rows /tmp/docvqa/rows_0_100.json --out /tmp/docvqa/run_rules
```

默认就读这两个路径。文件不在时，命令会说明没有编造金标。

## 在有显卡的机器上用本地 Qwen3-8B 测 DocVQA 2026

脚本是 `scripts/qwen3_8b_docvqa2026.py`。Qwen3-8B 是文本模型，不能看图。输入是已经做好的 OCR 页面文本，不是 PNG。本仓库不带权重，也不带验证集。不要在没有显卡的机器上下载或运行这个模型。

准备：

- 本地 Qwen3-8B 文本权重目录，用 `--weights` 传入。视觉版权重会被拒绝。
- 官方评测文件：克隆 [VLR-CVC/DocVQA2026](https://github.com/VLR-CVC/DocVQA2026)，把其中的 `eval_utils.py` 传给 `--eval-utils`。计分调用该文件里的 `evaluate_docvqa_prediction`，不在这里重写规则。
- 验证集 parquet：Hugging Face `VLR-CVC/DocVQA-2026` 的 `val.parquet`，传给 `--parquet`。脚本只读 `doc_id`、`doc_category`、`questions`、`answers`，不读页面图像。
- OCR 文本：`--ocr-root` 下每个 `doc_id` 一个目录，页文件名为 `p0000.txt`、`p0001.txt`，按文件名排序。

模型输出里必须有字面量 `FINAL ANSWER:`。官方函数会取最后一次出现之后的文本来和金标比较。没有这个标记就判错。脚本不会事后补上这个前缀。

有显卡的机器上先另装依赖，再运行：

```bash
.venv/bin/pip install torch transformers accelerate pyarrow python-Levenshtein python-dateutil
.venv/bin/python scripts/qwen3_8b_docvqa2026.py \
  --weights /path/to/Qwen3-8B \
  --eval-utils /path/to/DocVQA2026/eval_utils.py \
  --parquet /path/to/val.parquet \
  --ocr-root /path/to/ocr \
  --out runs/qwen3-8b-docvqa2026
```

`--max-ocr-chars` 默认 24000，超长 OCR 会截断并在结果里标明。`--limit` 只跑前 N 问。脚本强制离线加载权重。

终端打印 `Accuracy 判对数/问数`，以及各领域的同样计数。这个 Accuracy 只来自官方函数。它不是 `eval` 或 `recall-docvqa` 的哈希嵌入召回。
