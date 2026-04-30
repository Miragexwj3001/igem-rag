# iGEM Navigator

一个面向 iGEM 场景的专题智能检索与推理系统。

本项目基于历年 iGEM Wiki 数据、结构化元数据和 RAG 技术，构建了一个集 `首页洞察`、`知识检索`、`推理探索` 于一体的交互式应用，目标是为参赛队伍、老师和对合成生物学感兴趣的用户提供更准确、更可追溯的问答与分析能力。

## 项目简介

iGEM 每年都会沉淀大量公开 Wiki 页面，包含项目设计、实验方案、建模思路、人类实践、赛道方向等高价值内容。但这些信息往往分散在不同队伍页面中，人工阅读成本高，直接让通用大模型回答又容易出现幻觉。

因此，本项目尝试通过以下方式提升回答质量：

- 使用历年 iGEM Wiki 内容构建本地知识库
- 结合结构化表格数据进行筛选和增强
- 使用 RAG 检索相关证据，再让大模型基于证据回答
- 在问答之外，进一步支持面向新项目的推理与方案探索

## 当前功能

本项目目前提供 3 个主要页面：

### 1. 首页洞察

- 以卡片形式提供预设高价值问题
- 支持快速查看某类赛道、热门方向、代表项目等信息
- 适合第一次使用时快速了解系统能力

### 2. 往届项目知识检索台

- 支持按关键词、年份、赛道、奖项等条件过滤项目
- 支持连续对话式提问
- 支持锁定某个具体项目后继续追问
- 支持相似项目推荐
- 支持中文提问中文回答、英文提问英文回答
- 回答附带证据来源，便于回溯

### 3. 推理探索

- 面向新队伍输入项目设想
- 先召回相似历史项目和相关证据
- 再由大模型归纳生成可执行建议
- 输出内容包括：
  - 一句话结论
  - 三条执行主线
  - 8 周执行计划
  - 接下来 7 天的行动清单
  - 风险与缓解建议

## 技术路线

### 数据层

- 历年 iGEM Wiki 页面切块数据
- 向量索引：`out_embedding/vector_index.faiss`
- 文本元数据：`out_embedding/chunk_metadata.json`
- 结构化表格：`out_embedding/igem_teams.csv`

### 检索层

项目采用混合检索策略：

- `BM25`：负责关键词召回
- `FAISS + 向量嵌入`：负责语义召回
- `Cross-Encoder`：负责候选结果精排

此外，知识检索页中的相似项目推荐使用：

- `TF-IDF`
- `cosine similarity`

### 生成层

项目当前使用阿里云 DashScope 兼容接口调用 `Qwen` 模型。

核心思路不是直接让模型裸答，而是：

1. 先检索与问题最相关的证据片段
2. 再将证据组织成 Prompt
3. 让模型在证据约束下回答问题或生成方案

这样可以在一定程度上降低大模型幻觉，并增强结果的可解释性。

### 后处理层

- 中英文自动适配
- 文本清洗与格式整理
- 答案与来源绑定
- 结构化方案结果清洗与补全

## 项目结构

```text
igem-rag/
├─ app.py                        # 首页
├─ QwenRAGSystemOptimized.py     # RAG 检索与大模型调用核心逻辑
├─ ui_core.py                    # 共享 UI、样式、数据加载与工具函数
├─ requirements.txt              # Python 依赖
├─ README.md
├─ out_embedding/
│  ├─ vector_index.faiss
│  ├─ chunk_metadata.json
│  ├─ igem_teams.csv
│  └─ processing_stats.json
└─ pages/
   ├─ 1_知识检索台.py
   └─ 2_推理探索.py
```

## 环境要求

建议环境：

- Python 3.10 左右
- Windows / macOS / Linux 均可
- 建议使用虚拟环境

## 安装依赖

在项目根目录运行：

```powershell
pip install -r requirements.txt
```

注意：

项目代码中使用了 `langdetect` 做语言识别，但当前 `requirements.txt` 中未显式列出，因此建议额外安装：

```powershell
pip install langdetect
```

## 数据准备

运行前请确认 `out_embedding` 目录存在以下文件：

- `vector_index.faiss`
- `chunk_metadata.json`
- `igem_teams.csv`

本项目当前仓库中已经包含这些文件。

## API Key 配置

项目依赖 DashScope API Key 调用 Qwen 模型。

有两种方式提供：

### 方式一：页面内输入

启动后可直接在左侧栏输入 `DASHSCOPE_API_KEY`。

### 方式二：设置环境变量

PowerShell 示例：

```powershell
$env:DASHSCOPE_API_KEY="你的API_KEY"
```

## 运行方式

在项目根目录执行：

```powershell
streamlit run app.py
```

正常情况下，启动后可在浏览器中访问：

```text
http://localhost:8501
```

## 使用说明

### 首页

- 点击卡片可直接触发预设问题
- 支持输入自定义问题

### 知识检索台

- 先在左侧用年份、赛道、关键词等条件缩小范围
- 可选择某个项目并“锁定项目”
- 再在右侧聊天区连续提问

### 推理探索

- 输入你的项目简介
- 补充队伍优势、预算等级、强调方向
- 系统将自动检索相似项目并生成执行建议

## 项目亮点

- 面向 iGEM 垂直场景，而不是通用问答
- 结合了非结构化 Wiki 文本和结构化元数据
- 使用混合检索和重排序提升证据召回质量
- 不只做检索问答，还尝试做项目规划与探索
- 回答可追溯到来源项目，提高可信度

## 当前局限

- 数据仍主要来自 Wiki 页面，完整性受原始页面质量影响
- 奖项与官方结果类问题未必都能从 Wiki 中直接检索到
- 第 3 页的“推理探索”仍属于辅助规划，不等同于自动科研设计
- 一些模型和检索组件较重，首次加载可能较慢

## 后续可扩展方向

- 进一步完善 metadata 结构和标签体系
- 补充官方奖项、赛道、队伍信息
- 引入更细粒度的项目画像与冲奖策略分析
- 针对合成生物学语料做专门微调或偏好优化
- 引入更完整的 Agent 工作流

## 致谢

感谢 iGEM 历年公开 Wiki 页面提供的数据基础，也感谢相关开源工具生态，包括：

- Streamlit
- FAISS
- sentence-transformers
- scikit-learn
- rank-bm25
- Qwen / DashScope API

## 说明

本项目更适合作为：

- iGEM 历史资料智能检索工具
- 竞赛辅助分析平台
- 合成生物学专题问答与探索原型系统

如果你希望继续扩展本项目，建议优先从 `QwenRAGSystemOptimized.py`、`ui_core.py` 和 `pages/` 目录开始阅读。
