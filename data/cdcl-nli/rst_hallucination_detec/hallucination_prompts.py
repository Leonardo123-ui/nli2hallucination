"""
LLM 幻觉检测评估 - Prompt 模板库
支持三种不同的输入级别：
1. 仅输入：context + output
2. 增强输入：context + output + RST-RGAT 全局判别结果
3. 完整输入：context + output + RST-RGAT 所有结果（全局 + EDU 标签）
"""


class HallucinationPrompts:
    """幻觉检测 prompt 模板集合"""

    @staticmethod
    def prompt_type_1_llm_only(context: str, output: str) -> str:
        """
        Prompt 类型 1：仅基于 context 和 output 判断
        不提供任何 RST-RGAT 信息
        """
        prompt = f"""你是一个专业的幻觉检测分析师。

【任务】根据原始文章和生成摘要，判断摘要中是否存在幻觉（与原文不符的内容）。

【待分析文本】
原始文章：
{context}

生成摘要：
{output}

请完成以下任务：
1. 【验证】判断摘要中是否存在幻觉（是/否/部分准确），并说明理由
2. 【定位】指出摘要中具体哪些内容与原文不符
3. 【分析】解释产生幻觉的原因（事实冲突 / 无中生有 / 过度推断 / 时间混淆等）
4. 【建议】给出修正后的摘要"""
        return prompt

    @staticmethod
    def prompt_type_2_with_rst_global(
        context: str,
        output: str,
        rst_hallucination: bool,
        rst_prob: float
    ) -> str:
        """
        Prompt 类型 2：基于 context、output 和 RST-RGAT 全局判别结果
        提供 RST-RGAT 模型的全局幻觉检测结果，但不提供 EDU 级别的细节
        """
        rst_verdict = "发现幻觉" if rst_hallucination else "未发现幻觉"

        prompt = f"""你是一个专业的幻觉检测分析师。

【任务】根据原始文章、生成摘要和自动检测结果，独立判断摘要是否存在幻觉。

【待分析文本】
原始文章：
{context}

生成摘要：
{output}

【参考信息】
自动检测模型意见：{rst_verdict}（置信度：{rst_prob:.1%}）

【关键指示】
- 自动检测结果仅供参考，请**完全独立地核实**摘要是否与原文事实相符
- 如果自动检测与你的独立判断矛盾，优先采用你的判断

【最终输出】请在回复开头立即给出：
【最终结论】0（无幻觉）或 1（存在幻觉）

然后详细说明理由：
1. 摘要与原文是否存在事实矛盾或遗漏
2. 如果存在幻觉，具体指出位置和类型（事实冲突/无中生有/过度推断/时间混淆/因果错误等）
3. 修正建议"""
        return prompt

    @staticmethod
    def prompt_type_3_with_rst_full(
        context: str,
        output: str,
        rst_hallucination: bool,
        rst_prob: float,
        hallucination_edus: list
    ) -> str:
        """
        Prompt 类型 3 优化版（Type 3.1）：采用双盲核查法 + 反证逻辑强制链

        设计思路：
        1. 双盲核查：先独立分析，再对比小模型建议（解耦认知依赖）
        2. 反证强制链：判幻觉必须找到原文直接矛盾证据（提升Precision）
        3. 消除模糊：禁用"部分准确"，严格的0/1判定（硬化标签）
        """
        rst_verdict = "发现幻觉" if rst_hallucination else "未发现幻觉"

        # 构建高置信度EDU信息
        edu_info = ""
        if hallucination_edus:
            edu_info = "\n【高风险争议点（仅供对比参考，需独立验证）】\n"
            for i, edu in enumerate(hallucination_edus, 1):
                edu_text = edu.get('edu_text', '') if isinstance(edu, dict) else str(edu)
                prob = edu.get('prob', 0.0) if isinstance(edu, dict) else 0.0
                edu_info += f"{i}. {edu_text}（检测置信度：{prob:.1%}）\n"
        else:
            edu_info = "\n【系统信息】未发现高置信度的争议点"

        prompt = f"""你是一位严谨的事实核查员。你的任务是核对摘要是否相对于原文存在幻觉。

【原始文章】
{context}

【生成摘要】
{output}

【执行步骤】

**阶段 A - 独立盲审（忽略以下系统建议，先自行分析）：**
1. 逐行阅读摘要，识别其中3个关键事实性陈述
2. 在原文中逐一核实这些陈述的支撑证据
3. 如果某个陈述在原文中找不到直接支持，记录为"潜在风险点"

**阶段 B - 对比校验（与系统建议进行交叉验证）：**
系统全局检测：{rst_verdict}（置信度：{rst_prob:.1%}）
{edu_info}

4. 检查系统列出的争议点是否与你的"潜在风险点"有交集
5. 对于交集部分，重新严格审视原文，确认是否真的存在矛盾

【判幻觉的充要条件】
只有同时满足以下条件，才判定为"有幻觉（1）"：
✓ 摘要中存在原文**明确不支持**的陈述
✓ 你能在原文中**直接引述**与摘要矛盾的句子
✓ 这种矛盾构成**事实冲突**（不是仅仅"未提及"或"推理差异"）

如果仅是"原文未提及但不违背常识"或"表述方式不同但逻辑一致"，判定为"无幻觉（0）"。

【禁止条件】
❌ 禁止使用"部分准确"、"基本符合"等模糊措辞
❌ 不要因为系统的预警就改变自己的判断
❌ 不要因为细微措辞差异就判定为幻觉

【输出格式】
请在回复开头立即给出：
【结论】0（无幻觉）或 1（存在幻觉）

然后输出分析：
1. 【核实结果】：你的3个关键事实在原文中的支撑情况
2. 【对比发现】：与系统建议的交集及最终判断
3. 【反证证据】：（仅当判为幻觉时）原文中的直接矛盾句子
4. 【修正建议】：（可选）改写建议"""
        return prompt


# 快速访问函数
def get_prompt(
    prompt_type: int,
    context: str,
    output: str,
    rst_hallucination: bool = None,
    rst_prob: float = None,
    hallucination_edus: list = None
) -> str:
    """
    根据类型返回相应的 prompt

    Args:
        prompt_type: 1, 2, 或 3
        context: 原始文章
        output: 生成摘要
        rst_hallucination: RST-RGAT 全局判别（prompt_type >= 2 需要）
        rst_prob: RST-RGAT 置信度（prompt_type >= 2 需要）
        hallucination_edus: EDU 标签列表（prompt_type == 3 需要）

    Returns:
        对应的 prompt 字符串
    """
    if prompt_type == 1:
        return HallucinationPrompts.prompt_type_1_llm_only(context, output)
    elif prompt_type == 2:
        return HallucinationPrompts.prompt_type_2_with_rst_global(
            context, output, rst_hallucination, rst_prob
        )
    elif prompt_type == 3:
        return HallucinationPrompts.prompt_type_3_with_rst_full(
            context, output, rst_hallucination, rst_prob, hallucination_edus or []
        )
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 1, 2, or 3.")


# 使用示例
if __name__ == "__main__":
    # 示例文本
    context = "中国是世界上人口最多的国家，有超过14亿人。首都是北京。"
    output = "中国有20亿人，首都是上海。"

    # Type 1: 仅 LLM
    print("=" * 80)
    print("Prompt Type 1: LLM Only")
    print("=" * 80)
    prompt1 = get_prompt(1, context, output)
    print(prompt1)

    # Type 2: 含全局结果
    print("\n" + "=" * 80)
    print("Prompt Type 2: With RST Global Result")
    print("=" * 80)
    prompt2 = get_prompt(
        2, context, output,
        rst_hallucination=True,
        rst_prob=0.85
    )
    print(prompt2)

    # Type 3: 含完整结果
    print("\n" + "=" * 80)
    print("Prompt Type 3: With RST Full Results (including EDUs)")
    print("=" * 80)
    prompt3 = get_prompt(
        3, context, output,
        rst_hallucination=True,
        rst_prob=0.85,
        hallucination_edus=[
            {'edu_text': '中国有20亿人', 'is_hallucination': True, 'prob': 0.92},
            {'edu_text': '首都是上海', 'is_hallucination': True, 'prob': 0.88}
        ]
    )
    print(prompt3)
