# 多模态 PDF Agent

用 LangChain 读一份 PDF，加上可选的 BTOM 运行 JSON。入库顺序是固定的：分页、对图片页做 OCR、切块、写入 Chroma。提问才进入工具调用。

气象实验复核不在这个项目里。BTOM 的归因数学也没有改。

## 能做的两件事

- 核对注意力图和角色账本。图上的 `L3H7 / token:V` 只和检索到的账本原句比较，不对着原始账本列表再扫一遍。
- 把网格图写成候选 A/B/C、Query、Output 的整数 JSON。色号名字只来自检索到的规则原文。格子里的 10 会丢掉，不进入答案。

没有检索命中时，回答是「证据不足」。

- 审计注意力柱状图和该页结论文字是否一致。结论句从 PDF 文本层读取。最高柱的 key 下标按每张图自己的 x 轴刻度短线标定，取最高蓝柱中心。像素上一样高的柱保留两个下标，不猜。只有唯一的图下标和结论不同，才记为不一致。这一步不读 `true_peak`，也不读金标文件。

```bash
.venv/bin/python -m btom_pdf_agent audit-consistency chart.pdf
.venv/bin/python -m btom_pdf_agent audit-consistency chart.pdf --gold gold_inconsistent_ids.json
```

`--gold` 只在预测完成之后打开，用来算不一致页的集合召回。这一次 500 页上，不一致页召回是 200/200，误报 0 页。像素并列的 `p0152` 保留两个下标，没有判成不一致。这不是 499/500：那是最高柱下标和记录峰值是否一致的可行性核对。

## 技术边界

- 提问循环是 `langchain.agents.create_agent`。LangChain 1.4 的这个入口会把工具循环编译成图来执行。入库链是单独的 `Runnable` 序列，不让模型决定分页顺序。
- OCR 接口按 [RapidOCR](https://github.com/RapidAI/RapidOCR) 的调用方式留了 `--ocr rapidocr`。测试默认是确定性假引擎，只读本仓库用字格画出来的图，不下载 ONNX，也不写识别准确率。
- 检索是 Chroma 向量加 BM25，再用倒数排名合并。这不是神经网络重排序。向量默认是哈希嵌入，不下载权重。小块只用来检索，返回的是它所属的父块全文。
- 切块按标题或空行段落切开，过短的段会合并，不跨过标题，也不调用模型。页面上已有的页码、正文或 OCR、标题会写进块的元数据，检索可以按这三个字段过滤。
- 查询改写是本地规则：问句里只有一个明确所指时才补代词，出现两个问号才拆开，另外用一张很小的同义词表。不访问网络，不读 API key。没改写的原问句仍然参加检索。
- 两条已经检索到的段落如果对所问事实给出不同的值，回答是「证据不足」，不从中挑一条。检索为空时同样是「证据不足」。回答要带出处。
- Prompt 在 `prompts/v1.md`，版本号是 `v1`，里面有两条例子。
- 工具失败会再试一次。嵌入按文本哈希缓存在入库目录。回答可以走流式回调；测试不访问外网。
- 模型密钥只读 `OPENAI_API_KEY`，可选 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。没有 key 时用脚本模型把工具顺序跑通。

不要写成已经做到的：Multi-Agent、vLLM、Ollama、Milvus、Elasticsearch、神经重排、GraphRAG、用模型做语义切分、SFT、RL、知识飞轮、微调效果、微服务、可以写进简历的准确率。

## 运行

```bash
cd pdf-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/python -m btom_pdf_agent eval
```

`eval` 会打印这一次夹具的引用命中条数，例如 `夹具引用命中 3/3`。那是这三道夹具题的结果，不是准确率。

检索 Recall@K 的 K 仍是 `HybridIndex.search` 的默认返回条数（这一版是 4）。`data_uniform/Hi_ToM_order_1.csv` 的每一行只把故事正文写成 PDF 的一页，问句、答案和选项都不写入页面，因此问句不在被索引的页面里。查询是 CSV 的 question。金标是该行故事页返回的父块。若问句字符串出现在这些金标块里，这一行丢弃，不计入分母。命中是指该题每一个金标父块 id 都出现在前 K 条里。引用命中是脚本回答里写出了这些金标块 id，不要求问句本身出现在块文本中。IOI 在克隆里没有数据文件，ARC 也没有能让 `export_mtm_csv.py` 在不下载权重时写出的 challenges JSON，所以这两项跳过，也不另造数据。

DocVQA 文本检索另走 `python -m btom_pdf_agent recall-docvqa`。用的是 `nielsr/docvqa_1200_examples` 前 100 问和数据集自带文本，不跑 RapidOCR，也不算 ANLS。金标父块是子块范围盖住证据片段的那些父块。

评测会多打一行「金标被切散」。它只表示金标文本有没有落在多于一个检索小块上，不是 MRR，也不是准确率。Hi-ToM 的每一页故事仍然是一个父块，整段返回；100/100 是说故事被切成了多个小块去检索。

下面的比例只属于这一次运行，不能写成稳定成绩：

| 集 | 上一次 | 这一次 | 金标被切散 |
| --- | --- | --- | --- |
| Hi-ToM 故事页 Recall@4 | 99/100 | 97/100 | 100/100 |
| DocVQA 文本检索 Recall@4 | 59/100 | 63/100 | 5/100 |

Hi-ToM 这一次的引用命中是 97/100，和召回是同一批题。夹具三题的引用检查仍是 `夹具引用命中 3/3`，那不是准确率。

自己的 PDF：

```bash
.venv/bin/python -m btom_pdf_agent ingest your.pdf --state examples/run_state.json --out runs/demo
.venv/bin/python -m btom_pdf_agent ask runs/demo "请核对第 2 页注意力图和账本" --fake
```

有 API key 时去掉 `--fake` 才会请求模型。图片页需要能被 pypdf 抽出内嵌图。文本页走文本层，不经过 OCR。
