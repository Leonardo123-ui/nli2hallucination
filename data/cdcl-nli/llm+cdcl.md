# 任务说明：基于 LLM 增强的长文本幻觉判别器

## 1. 目标 (Objective)
实现一个通过 **"Rationale Injection" (理由注入)** 策略，结合 **生成式 LLM (API调用)** 和 **本地小模型 (具备长文本能力)** 的幻觉检测流水线。
核心思想：利用 LLM 生成“可疑点分析” (Rationale)，将其注入到本地小模型的输入中，引导其注意力机制，从而发挥小模型处理长文本的优势。

## 2. 架构流程 (Architecture Flow)
1.  **输入:** `Source_Text` (原文), `Summary` (摘要)
2.  **阶段 1 (LLM - 分析员):**
    * 将 `Source` + `Summary` 发送给 LLM。
    * 任务：识别潜在的不一致或幻觉。
    * 输出：`LLM_Rationale` (一段简短的文本，描述可疑点)。
3.  **阶段 2 (拼接 - Concatenation):**
    * 构造最终输入字符串：`[CLS] LLM_Rationale [SEP] Summary [SEP] Source_Text` (请根据 Tokenizer 实际的分隔符进行适配)。
4.  **阶段 3 (小模型 - 判官):**
    * 将拼接后的长文本喂给本地判别器。
    * 输出：最终的概率/标签 (Score/Label)。

## 3. 实现要求 (Implementation Requirements)

### 技术栈
* Python 3.x
* `transformers` & `torch` (用于加载本地模型)

### 文件结构 (保持扁平与最小化)
* `detector.py`: 包含核心类 `AugmentedDiscriminator`。
* `main.py`: 示例调用脚本。
* `requirements.txt`: 最小依赖列表。

### 详细逻辑细节

#### A. LLM 组件 (`detector.py`)
创建一个方法来调用 LLM。使用专门用于 **分析 (Analysis)** 而非最终判决的 System Prompt。
* **System Prompt:** "你是一名幻觉侦探。请简要列出 Summary 和 Source 之间的任何不一致之处。如果没有，请回答 'No inconsistencies found'。请将字数控制在 50 字以内。"
* **输入:** Source + Summary。
* **输出:** 分析文本 (Rationale)。

#### B. 判别器类 (`detector.py`)
该类应初始化本地模型（假设使用 HuggingFace 路径）和 LLM 客户端。
* 方法 `predict(source, summary)`:
    1.  调用 LLM 获取 `rationale`。
    2.  格式化输入: `{rationale} </s> {summary} </s> {source}` (使用 tokenizer 特定的分隔符)。
    3.  Tokenize 并运行本地模型推理。
    4.  返回分数 (Score)。

#### C. 清理工作 (`main.py`)
* `main.py` 应生成虚构的 `Source` 和 `Summary` 来运行一次完整的流水线测试。
* **关键要求:** 程序运行结束后，磁盘上**不得残留**任何 `.log`、`.cache` 或测试输出文件。如有中间存储需求请使用 `tempfile` 或全内存处理。

## 4. 完成标准 (Definition of Done)
1.  代码仅包含在上述请求的最小文件中。
2.  小模型的输入文本成功包含了 LLM 生成的 Rationale。
3.  脚本运行无错并打印出最终分数。
4.  **无垃圾文件残留**。

## 5. 约束条件 (Constraints)
* **不要**生成复杂的目录结构。
* **不要**添加不必要的日志记录或可视化工具。
* 假设用户已经准备好了本地模型路径和 API Key（代码中使用占位符即可）。