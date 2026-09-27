# PDF Agent 用法

在 `pdf-agent` 目录里用命令行运行。这个子目录没有 HTTP 服务，也不要把它说成气象项目的 `uvicorn app:app`。

检索默认用哈希嵌入。`eval` 和 `recall-docvqa` 打出来的召回是这一次检索的计数，不是 Qwen 的成绩。

本地 Qwen3-8B 的主路径是下面这份 DocVQA 1000 问，计分是 ANLS。那不是 DocVQA 2026，也不是检索召回。

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

这是另一条已经存在的命令，做的是旧 DocVQA 文本子集上的哈希嵌入检索 Recall。它不看图，不算 ANLS，也不是 Qwen 的成绩。此前在全部 1000 问上的 Recall@4 是 554/1000，那个数是检索召回，不要把它当成 ANLS。

```bash
.venv/bin/python -m btom_pdf_agent recall-docvqa --rows /tmp/docvqa/rows_0_100.json --out /tmp/docvqa/run_rules
```

默认就读这两个路径，用的是前 100 问。文件不在时，命令会说明没有编造金标。

## 在有显卡的机器上用本地 Qwen3-8B 测 DocVQA（1000 问）

这是主路径。数据是 Hugging Face `nielsr/docvqa_1200_examples` 的 train 全部 1000 问，和算出检索 Recall@4 554/1000 的是同一批问题。不是官方 DocVQA 验证集，也不是 DocVQA 2026。

数据已经在仓库里：

`pdf-agent/docvqa/nielsr_docvqa_1200_examples_train.jsonl`

每一行有 `id`、英文 `question`、金标 `answers`、这一页的 `page_text`（数据集里的词用空格连起来）。没有页面图片。Qwen3-8B 是文本模型，不能看图。脚本只把 `page_text` 送给模型。

554/1000 是当时哈希嵌入检索的 Recall@4。下面这个脚本打印的是 ANLS，两件事不能混用。

ANLS 的算法：送进公式的模型答案，是新生成文本去掉 `<think>` 块、再去掉首尾空白之后的部分。对这个答案和每一个金标，先转成小写并去掉首尾空白，再算归一化编辑相似度。编辑距离记为 Lev，归一化距离 NL = Lev / max(两边长度)；两边都是空串时 NL = 0。NL < 0.5 时，相似度是 1 − NL，否则这个金标记 0。一题取各金标相似度里的最大值。ANLS 是全部问题这个最大值的平均。

有显卡的机器上先另装依赖。`--weights` 指向本机已经放好的 Qwen3-8B 文本权重目录。视觉权重会直接拒绝。脚本强制离线，不会下载模型。不要在没有显卡的机器上下载或运行 Qwen。

```bash
.venv/bin/pip install torch transformers accelerate
.venv/bin/python scripts/qwen3_8b_docvqa.py \
  --weights /path/to/Qwen3-8B \
  --out runs/qwen3-8b-docvqa
```

数据路径不用再传，脚本默认读上面那个 jsonl。`--limit` 只跑前 N 问。`--max-page-chars` 默认 24000；这份数据最长一页不到 8000 字符，默认不会截断。终端打印 `ANLS 小数（问数）`。

## 不建议用 8B 测 DocVQA 2026

DocVQA 2026 要看页面图像，官方计分是 Accuracy，不是上面的 ANLS。Qwen3-8B 不能看图，不建议用它测 2026。仓库里不放 2026 的 parquet，也不放页面图片。`scripts/qwen3_8b_docvqa2026.py` 还在，但那不是这条 1000 问的主路径。
