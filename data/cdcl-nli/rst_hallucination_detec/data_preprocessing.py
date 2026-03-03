"""
数据预处理模块
功能：
1. 加载summary_data.json
2. 使用DMRST解析context和output
3. EDU字符对齐
4. 标签对齐（hallucination_labels → EDU标签）
"""

import json
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 提前添加DMRST模块路径
DMRST_PATH = "/mnt/nlp/yuanmengying/CDCL_NLI-old"
if DMRST_PATH not in sys.path:
    sys.path.insert(0, DMRST_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DMRSTProcessor:
    """DMRST解析器封装"""

    def __init__(self):
        # 导入DMRST模块
        try:
            from RST_Parser.DM_RST import RST_Tree, precess_rst_result
            self.rst_tree = RST_Tree()
            self.model = self.rst_tree.init_model()
            self.precess_rst_result = precess_rst_result()  # 实例化类
            logger.info("✓ DMRST模型加载完成")
        except ImportError as e:
            logger.error(f"DMRST导入失败: {e}")
            logger.error(f"请确保 {DMRST_PATH} 路径存在且包含 RST_Parser/DM_RST.py")
            raise

    def parse_text(self, text: str) -> Dict:
        """
        解析单个文本

        Returns:
            {
                'edu_breaks': [5, 10, ...],  # Token索引
                'tree': '(1:Satellite=Contrast:4, 5:Nucleus=span:6)',
                'segments': ['EDU1', 'EDU2', ...],  # EDU文本
                'token_offsets': [(0, 5), (6, 10), ...]  # 字符偏移
            }
        """
        input_sentences = [text]

        # DMRST推理
        (
            input_sentences_batch,
            all_segmentation_pred_batch,
            all_tree_parsing_pred_batch,
        ) = self.rst_tree.inference(self.model, input_sentences)

        # 获取EDU分割
        segments = self.precess_rst_result.merge_strings(
            input_sentences_batch[0],
            all_segmentation_pred_batch[0]
        )

        # 获取RST树
        rst_info = all_tree_parsing_pred_batch[0][0].split()

        # 提取EDU breaks（Token索引）
        edu_breaks = all_segmentation_pred_batch[0]

        # 计算字符偏移量
        token_offsets = self._compute_char_offsets(text, segments)

        return {
            'edu_breaks': edu_breaks,
            'tree': ' '.join(rst_info),
            'segments': segments,
            'token_offsets': token_offsets
        }

    def _compute_char_offsets(self, text: str, segments: List[str]) -> List[Tuple[int, int]]:
        """
        计算每个EDU的字符偏移量

        Args:
            text: 原始文本
            segments: EDU文本列表

        Returns:
            [(start, end), ...] 每个EDU的字符范围
        """
        offsets = []
        current_pos = 0

        for seg in segments:
            seg_clean = seg.strip()
            if not seg_clean:
                continue

            # 在文本中查找这个EDU
            start = text.find(seg_clean, current_pos)
            if start == -1:
                # 如果找不到，尝试部分匹配
                start = current_pos

            end = start + len(seg_clean)
            offsets.append((start, end))
            current_pos = end

        return offsets


class LabelAligner:
    """标签对齐器"""

    @staticmethod
    def align_labels_to_edus(
        edu_offsets: List[Tuple[int, int]],
        hallucination_labels: List[Dict]
    ) -> List[int]:
        """
        将字符级标签对齐到EDU级

        判定逻辑：如果EDU和幻觉Span有重叠，则该EDU标记为1

        Args:
            edu_offsets: [(start, end), ...] EDU的字符范围
            hallucination_labels: [{'start': 10, 'end': 25, 'label_type': ...}, ...]

        Returns:
            [0, 0, 1, 1, 0, ...] 每个EDU的标签
        """
        edu_labels = []

        for e_start, e_end in edu_offsets:
            is_hallucination = 0

            # 检查是否与任何幻觉Span重叠
            for label in hallucination_labels:
                h_start = label.get('start', 0)
                h_end = label.get('end', 0)

                # 重叠判定：max(e_start, h_start) < min(e_end, h_end)
                if max(e_start, h_start) < min(e_end, h_end):
                    is_hallucination = 1
                    break

            edu_labels.append(is_hallucination)

        return edu_labels


class DataPreprocessor:
    """数据预处理主类"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dmrst = DMRSTProcessor()
        self.aligner = LabelAligner()

    def load_data(self) -> List[Dict]:
        """加载原始数据"""
        logger.info(f"加载数据: {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ 加载完成，共 {len(data)} 条样本")
        return data

    def process_single_sample(self, sample: Dict) -> Dict:
        """
        处理单个样本

        Args:
            sample: {
                'id': '276',
                'context': 'Aaron Hernandez...',
                'output': 'Hernandez found guilty...',
                'hallucination_labels_processed': {...},
                'dataset': 'train' or 'test'
            }

        Returns:
            {
                'id': '276',
                'context': {...},  # DMRST解析结果
                'output': {...},   # DMRST解析结果 + EDU标签
                'y_global': 0/1,
                'dataset': 'train' or 'test'
            }
        """
        sample_id = sample.get('id', 'unknown')
        context = sample.get('context', '')
        output = sample.get('output', '')
        dataset_split = sample.get('dataset', 'train')  # 保留dataset字段

        # 解析hallucination_labels（使用原始列表，不是processed计数）
        hall_labels_raw = sample.get('hallucination_labels', [])

        # 如果是字符串（JSON编码），先解析
        if isinstance(hall_labels_raw, str):
            try:
                hall_labels_raw = json.loads(hall_labels_raw)
            except json.JSONDecodeError:
                logger.warning(f"样本 {sample_id} hallucination_labels解析失败")
                hall_labels_raw = []

        hallucination_labels = self._extract_hallucination_labels(hall_labels_raw)

        # DMRST解析
        try:
            context_parsed = self.dmrst.parse_text(context)
            output_parsed = self.dmrst.parse_text(output)
        except Exception as e:
            logger.warning(f"样本 {sample_id} DMRST解析失败: {e}")
            return None

        # 标签对齐（仅对output的EDU）
        output_edu_labels = self.aligner.align_labels_to_edus(
            output_parsed['token_offsets'],
            hallucination_labels
        )

        # 全局标签
        y_global = 1 if any(output_edu_labels) else 0

        return {
            'id': sample_id,
            'context': context_parsed,
            'output': output_parsed,
            'output_edu_labels': output_edu_labels,
            'y_global': y_global,
            'dataset': dataset_split  # 保留dataset字段
        }

    def _extract_hallucination_labels(self, labels_list: List) -> List[Dict]:
        """
        提取幻觉标签的span信息

        Args:
            labels_list: [
                {'start': 636, 'end': 653, 'text': '...', 'label_type': '...'},
                ...
            ]

        Returns:
            [{'start': 636, 'end': 653, 'label_type': 'Evident Conflict'}, ...]
        """
        all_spans = []

        if not isinstance(labels_list, list):
            logger.warning(f"hallucination_labels不是列表: {type(labels_list)}")
            return all_spans

        for span in labels_list:
            if isinstance(span, dict) and 'start' in span and 'end' in span:
                all_spans.append({
                    'start': span['start'],
                    'end': span['end'],
                    'label_type': span.get('label_type', 'unknown')
                })

        return all_spans

    def process_all(self, save_path: str = None, max_samples: int = None) -> List[Dict]:
        """
        处理所有样本

        Args:
            save_path: 保存处理后数据的路径
            max_samples: 最大处理样本数（用于测试）

        Returns:
            处理后的样本列表
        """
        raw_data = self.load_data()

        if max_samples:
            raw_data = raw_data[:max_samples]
            logger.info(f"仅处理前 {max_samples} 个样本")

        processed_data = []

        for i, sample in enumerate(raw_data):
            if (i + 1) % 100 == 0:
                logger.info(f"处理进度: {i + 1}/{len(raw_data)}")

            processed = self.process_single_sample(sample)
            if processed is not None:
                processed_data.append(processed)

        logger.info(f"✓ 处理完成，成功 {len(processed_data)}/{len(raw_data)} 个样本")

        # 保存
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 已保存到: {save_path}")

        return processed_data


def split_by_dataset_field(data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    根据dataset字段划分训练集和测试集

    Args:
        data: 处理后的数据（包含原始的dataset字段）

    Returns:
        (train_data, test_data)
    """
    train_data = [item for item in data if item.get('dataset', 'train') == 'train']
    test_data = [item for item in data if item.get('dataset', 'train') == 'test']

    logger.info(f"数据划分: 训练集 {len(train_data)}, 测试集 {len(test_data)}")

    return train_data, test_data


if __name__ == "__main__":
    # 测试数据预处理
    data_path = "/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json"

    preprocessor = DataPreprocessor(data_path)

    # 先处理前10个样本测试
    logger.info("开始测试数据预处理（前10个样本）...")
    processed_data = preprocessor.process_all(max_samples=10)

    # 显示统计信息
    if processed_data:
        sample = processed_data[0]
        logger.info(f"\n示例样本:")
        logger.info(f"  ID: {sample['id']}")
        logger.info(f"  Context EDUs: {len(sample['context']['segments'])}")
        logger.info(f"  Output EDUs: {len(sample['output']['segments'])}")
        logger.info(f"  Output EDU Labels: {sample['output_edu_labels']}")
        logger.info(f"  Global Label: {sample['y_global']}")

    logger.info("\n✓ 数据预处理模块测试完成！")
