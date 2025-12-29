"""
对比BERT分类器和LLM推理结果
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelComparison:
    """模型对比类"""

    def __init__(self, bert_results_dir="./test_results", llm_results_dir="./llm_results"):
        """初始化对比工具"""
        self.bert_dir = Path(bert_results_dir)
        self.llm_dir = Path(llm_results_dir)

    def load_bert_results(self) -> Tuple[pd.DataFrame, dict]:
        """加载BERT结果"""
        print("加载BERT结果...")

        # 加载详细预测结果
        predictions_file = self.bert_dir / "detailed_predictions.xlsx"
        if not predictions_file.exists():
            raise FileNotFoundError(f"找不到文件: {predictions_file}")

        bert_df = pd.read_excel(predictions_file)

        # 加载评估指标
        metrics_file = self.bert_dir / "test_results.json"
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                bert_metrics = json.load(f)
        else:
            bert_metrics = None

        return bert_df, bert_metrics

    def load_llm_results(self, model_name: str = "llama3") -> Tuple[pd.DataFrame, dict]:
        """加载LLM结果"""
        print(f"加载LLM({model_name})结果...")

        llm_model_dir = self.llm_dir / model_name
        if not llm_model_dir.exists():
            # 尝试其他可能的目录名
            possible_dirs = list(self.llm_dir.glob("*"))
            if possible_dirs:
                llm_model_dir = possible_dirs[0]
                print(f"使用目录: {llm_model_dir}")
            else:
                raise FileNotFoundError(f"找不到LLM结果目录: {self.llm_dir}")

        # 加载详细预测结果
        predictions_file = llm_model_dir / "llm_detailed_predictions.xlsx"
        if not predictions_file.exists():
            raise FileNotFoundError(f"找不到文件: {predictions_file}")

        llm_df = pd.read_excel(predictions_file)

        # 加载评估指标
        metrics_file = llm_model_dir / "llm_results.json"
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                llm_metrics = json.load(f)
        else:
            llm_metrics = None

        return llm_df, llm_metrics

    def merge_results(self, bert_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
        """合并BERT和LLM的结果"""
        print("合并结果...")

        # 将llm预测结果添加到bert结果
        bert_df['llm_prediction'] = None
        bert_df['llm_confidence'] = None

        # 使用id字段进行合并
        if 'id' in bert_df.columns and 'id' in llm_df.columns:
            llm_map = dict(zip(llm_df['id'], llm_df['llm_prediction']))
            conf_map = dict(zip(llm_df['id'], llm_df['llm_confidence']))

            for idx, row in bert_df.iterrows():
                sample_id = row['id']
                if sample_id in llm_map:
                    bert_df.loc[idx, 'llm_prediction'] = llm_map[sample_id]
                    bert_df.loc[idx, 'llm_confidence'] = conf_map[sample_id]
        else:
            # 如果没有id字段，按顺序对齐
            bert_df['llm_prediction'] = llm_df['llm_prediction'].values[:len(bert_df)]
            bert_df['llm_confidence'] = llm_df['llm_confidence'].values[:len(bert_df)]

        # 添加一致性标记
        bert_df['bert_correct'] = bert_df['predicted_label'] == bert_df['label']
        bert_df['llm_correct'] = bert_df['llm_prediction'] == bert_df['label']
        bert_df['both_correct'] = bert_df['bert_correct'] & bert_df['llm_correct']
        bert_df['both_wrong'] = (~bert_df['bert_correct']) & (~bert_df['llm_correct'])
        bert_df['disagreement'] = bert_df['predicted_label'] != bert_df['llm_prediction']

        return bert_df

    def calculate_comparison_metrics(self, comparison_df: pd.DataFrame) -> dict:
        """计算对比指标"""
        print("计算对比指标...")

        total = len(comparison_df)

        # 一致性指标
        agreement = (comparison_df['predicted_label'] == comparison_df['llm_prediction']).sum() / total
        both_correct = comparison_df['both_correct'].sum() / total
        both_wrong = comparison_df['both_wrong'].sum() / total
        bert_only_correct = (comparison_df['bert_correct'] & ~comparison_df['llm_correct']).sum() / total
        llm_only_correct = (~comparison_df['bert_correct'] & comparison_df['llm_correct']).sum() / total

        # 针对有幻觉类别的分析
        hallucination_mask = comparison_df['label'] == 1
        hallucination_df = comparison_df[hallucination_mask]

        if len(hallucination_df) > 0:
            hall_agreement = (hallucination_df['predicted_label'] == hallucination_df['llm_prediction']).sum() / len(
                hallucination_df)
            hall_bert_recall = hallucination_df['bert_correct'].sum() / len(hallucination_df)
            hall_llm_recall = hallucination_df['llm_correct'].sum() / len(hallucination_df)
        else:
            hall_agreement = 0
            hall_bert_recall = 0
            hall_llm_recall = 0

        # 针对无幻觉类别的分析
        no_hallucination_mask = comparison_df['label'] == 0
        no_hallucination_df = comparison_df[no_hallucination_mask]

        if len(no_hallucination_df) > 0:
            no_hall_agreement = (no_hallucination_df['predicted_label'] == no_hallucination_df['llm_prediction']).sum() / len(
                no_hallucination_df)
            no_hall_bert_recall = no_hallucination_df['bert_correct'].sum() / len(no_hallucination_df)
            no_hall_llm_recall = no_hallucination_df['llm_correct'].sum() / len(no_hallucination_df)
        else:
            no_hall_agreement = 0
            no_hall_bert_recall = 0
            no_hall_llm_recall = 0

        return {
            'total_samples': total,
            'agreement_rate': agreement,
            'both_correct_rate': both_correct,
            'both_wrong_rate': both_wrong,
            'bert_only_correct_rate': bert_only_correct,
            'llm_only_correct_rate': llm_only_correct,
            'hallucination_agreement': hall_agreement,
            'hallucination_bert_recall': hall_bert_recall,
            'hallucination_llm_recall': hall_llm_recall,
            'no_hallucination_agreement': no_hall_agreement,
            'no_hallucination_bert_recall': no_hall_bert_recall,
            'no_hallucination_llm_recall': no_hall_llm_recall,
        }

    def print_comparison_report(
        self,
        comparison_df: pd.DataFrame,
        bert_metrics: dict,
        llm_metrics: dict,
        comparison_metrics: dict
    ):
        """打印对比报告"""

        print("\n" + "="*80)
        print("BERT vs LLM 幻觉检测对比报告")
        print("="*80)

        print("\n📊 总体一致性分析:")
        print(f"样本总数: {comparison_metrics['total_samples']}")
        print(f"预测一致率: {comparison_metrics['agreement_rate']:.2%}")
        print(f"两个模型都正确: {comparison_metrics['both_correct_rate']:.2%}")
        print(f"两个模型都错误: {comparison_metrics['both_wrong_rate']:.2%}")
        print(f"仅BERT正确: {comparison_metrics['bert_only_correct_rate']:.2%}")
        print(f"仅LLM正确: {comparison_metrics['llm_only_correct_rate']:.2%}")

        print("\n🔍 有幻觉样本 (标签=1) 分析:")
        print(f"预测一致率: {comparison_metrics['hallucination_agreement']:.2%}")
        print(f"BERT召回率: {comparison_metrics['hallucination_bert_recall']:.2%}")
        print(f"LLM召回率: {comparison_metrics['hallucination_llm_recall']:.2%}")

        print("\n✅ 无幻觉样本 (标签=0) 分析:")
        print(f"预测一致率: {comparison_metrics['no_hallucination_agreement']:.2%}")
        print(f"BERT准确率: {comparison_metrics['no_hallucination_bert_recall']:.2%}")
        print(f"LLM准确率: {comparison_metrics['no_hallucination_llm_recall']:.2%}")

        print("\n📈 BERT性能指标:")
        if bert_metrics:
            metrics = bert_metrics['detailed_metrics']
            print(f"准确率: {metrics['accuracy']:.4f}")
            print(f"幻觉F1分数: {metrics['hallucination']['f1_score']:.4f}")
            print(f"幻觉召回率: {metrics['hallucination']['recall']:.4f}")

        print("\n📈 LLM性能指标:")
        if llm_metrics:
            metrics = llm_metrics['detailed_metrics']
            print(f"准确率: {metrics['accuracy']:.4f}")
            print(f"幻觉F1分数: {metrics['hallucination']['f1_score']:.4f}")
            print(f"幻觉召回率: {metrics['hallucination']['recall']:.4f}")

        print("\n" + "="*80)

    def save_comparison_results(self, comparison_df: pd.DataFrame, output_dir="./comparison_results"):
        """保存对比结果"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 保存完整对比结果
        output_file = Path(output_dir) / "bert_llm_comparison.xlsx"
        comparison_df.to_excel(output_file, index=False)
        print(f"对比结果已保存: {output_file}")

        # 保存错误分析
        disagreement_df = comparison_df[comparison_df['disagreement']]
        disagreement_file = Path(output_dir) / "disagreement_cases.xlsx"
        disagreement_df.to_excel(disagreement_file, index=False)
        print(f"分歧案例已保存: {disagreement_file}")

        # 保存两个模型都错误的情况
        both_wrong_df = comparison_df[comparison_df['both_wrong']]
        both_wrong_file = Path(output_dir) / "both_wrong_cases.xlsx"
        both_wrong_df.to_excel(both_wrong_file, index=False)
        print(f"两个模型都错误的情况已保存: {both_wrong_file}")

        # 保存一致性统计
        consistency_stats = {
            '分类': ['预测一致', '两个都正确', '两个都错误', '仅BERT正确', '仅LLM正确'],
            '数量': [
                (comparison_df['predicted_label'] == comparison_df['llm_prediction']).sum(),
                comparison_df['both_correct'].sum(),
                comparison_df['both_wrong'].sum(),
                (comparison_df['bert_correct'] & ~comparison_df['llm_correct']).sum(),
                (~comparison_df['bert_correct'] & comparison_df['llm_correct']).sum(),
            ]
        }
        stats_df = pd.DataFrame(consistency_stats)
        stats_file = Path(output_dir) / "consistency_statistics.xlsx"
        stats_df.to_excel(stats_file, index=False)
        print(f"一致性统计已保存: {stats_file}")

    def run_full_comparison(self, output_dir="./comparison_results"):
        """运行完整的对比分析"""
        # 加载结果
        bert_df, bert_metrics = self.load_bert_results()
        llm_df, llm_metrics = self.load_llm_results()

        # 合并结果
        comparison_df = self.merge_results(bert_df, llm_df)

        # 计算对比指标
        comparison_metrics = self.calculate_comparison_metrics(comparison_df)

        # 打印报告
        self.print_comparison_report(comparison_df, bert_metrics, llm_metrics, comparison_metrics)

        # 保存结果
        self.save_comparison_results(comparison_df, output_dir)

        return comparison_df, comparison_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='对比BERT和LLM的推理结果')
    parser.add_argument('--bert_dir', default='./test_results',
                       help='BERT结果目录')
    parser.add_argument('--llm_dir', default='./llm_results',
                       help='LLM结果目录')
    parser.add_argument('--output_dir', default='./comparison_results',
                       help='对比结果保存目录')

    args = parser.parse_args()

    try:
        # 创建对比工具
        comparator = ModelComparison(
            bert_results_dir=args.bert_dir,
            llm_results_dir=args.llm_dir
        )

        # 运行对比分析
        comparison_df, metrics = comparator.run_full_comparison(args.output_dir)

        print(f"\n✅ 对比分析完成！结果已保存到: {args.output_dir}")

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请确保已运行BERT和LLM的推理脚本")


if __name__ == "__main__":
    main()
