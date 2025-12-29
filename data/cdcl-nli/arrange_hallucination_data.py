"""
幻觉检测数据的 CDCL-NLI 适配脚本

改编自 CDCL-NLI/arrange_data_new.py
使用 RST 修辞结构树和图神经网络进行幻觉检测

主要功能：
1. 使用 DM-RST 模型将 context 和 output 转换为修辞结构树
2. 使用 ModernBERT 生成节点 embeddings
3. 计算词汇链（lexical chains）矩阵
4. 生成用于 CDCL-NLI 模型的数据结构
"""

import json
import torch
import numpy as np
import nltk
import os
import sys
import pickle
import glob
from pathlib import Path
from tqdm import tqdm

# 设置设备 (优先使用环境变量，否则使用GPU 0)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("未设置CUDA_VISIBLE_DEVICES，默认使用GPU 0")
else:
    print(f"使用GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

# 设置CUDA调试模式（帮助定位错误）
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 如果使用GPU，打印GPU信息
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 添加 CDCL-NLI 路径以使用 DM_RST 模块
sys.path.append("/mnt/nlp/yuanmengying/CDCL-NLI")
from data.DM_RST import RST_Tree, precess_rst_result

# Transformers
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset


def ensure_parent_directory(path_str):
    """
    确保父目录存在，不存在则创建

    Args:
        path_str: 文件或目录路径
    """
    path = Path(path_str)
    parent_dir = path.parent

    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建父目录: {parent_dir}")
    else:
        print(f"父目录已存在: {parent_dir}")

    return str(parent_dir)


def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    v1 = v1.reshape(1, -1)
    v2 = v2.reshape(1, -1)

    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def extract_node_features(embeddings_data, idx, prefix):
    """提取节点特征"""
    node_features = []
    for item in embeddings_data[idx][prefix]:
        node_id, embedding, text = item
        node_features.append((node_id, embedding, text))
    return node_features


class HallucinationDataProcessor:
    """
    幻觉检测数据处理器

    功能：
    1. 加载幻觉检测数据（JSON格式）
    2. 使用 RST 模型提取修辞结构
    3. 保存 RST 结果
    """

    def __init__(self, mode, save_dir, purpose):
        """
        Args:
            mode: 是否保存（True/False）
            save_dir: 保存根目录
            purpose: 用途（train/test）
        """
        self.save_dir = os.path.join(save_dir, purpose)
        self.rst_path = "rst_result.jsonl"
        self.save_or_not = mode

    def read_json_lines(self, file_path):
        """读取 JSONL 文件"""
        oridata = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line.strip())
                    oridata.append(record)
                except json.JSONDecodeError as e:
                    print(f"JSON 解码错误: {line}")
                    print(e)
        return oridata

    def load_json(self, json_path):
        """加载 JSON 文件"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_data_length(json_path):
        """获取数据长度"""
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"数据长度: {len(data)}")
        return len(data)

    def write_jsonl(self, path, data):
        """写入 JSONL 文件"""
        with open(path, "w", encoding="utf-8") as file:
            for record in data:
                json_record = json.dumps(record, ensure_ascii=False)
                file.write(json_record + "\n")
        print(f"成功保存到 {path}")

    def get_tree(self, a):
        """
        将从 DM-RST 提取的节点结果转换为树结构

        Args:
            a: node_number

        Returns:
            tree: 树表示
            leaf_node: 叶子节点索引
            parent_dict: 父节点字典
        """
        tree = []
        list_new = [elem for sublist in a for elem in (sublist[:2], sublist[2:])]
        parent_node = [1]  # 根节点
        parent_dict = {}
        leaf_node = []

        for index, i in enumerate(list_new):
            if i[0] == i[1]:
                leaf_node.append(index + 2)
            else:
                parent_node.append(index + 2)
                key = str(i[0]) + "_" + str(i[1])
                parent_dict[key] = index + 2

            if index < 2:
                tree.append([1, index + 2])

        for index, j in enumerate(a):
            if index == 0:
                continue
            else:
                key = str(j[0]) + "_" + str(j[3])
                parent = parent_dict[key]
                tree.append([parent, (index + 1) * 2])
                tree.append([parent, (index + 1) * 2 + 1])

        return parent_dict, leaf_node, tree

    def get_rst(self, data, rst_results_store_path):
        """
        对 context 和 output 进行 RST 分析

        Args:
            data: 幻觉检测数据
            rst_results_store_path: RST 结果保存路径

        Returns:
            rst_results: RST 分析结果列表
        """
        ensure_parent_directory(rst_results_store_path)
        print(f"RST 结果路径: {rst_results_store_path}")

        if os.path.exists(rst_results_store_path):
            rst_results = self.get_stored_rst(rst_results_store_path)
            print("已存在 RST 结果")
            return rst_results

        my_rst_tree = RST_Tree()
        model = my_rst_tree.init_model()
        precess_rst_tree = precess_rst_result()

        batch_size = 100
        rst_results = []
        count = 0

        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]

            # 批量构建输入句子列表
            input_sentences = []
            for item in batch_data:
                context = item["news1_origin"]    # context 作为 premise
                output = item["news2_origin"]     # output 作为 hypothesis
                input_sentences.append(context)
                input_sentences.append(output)

            # 批量推理
            (
                input_sentences_batch,
                all_segmentation_pred_batch,
                all_tree_parsing_pred_batch,
            ) = my_rst_tree.inference(model, input_sentences)

            for index, i in enumerate(batch_data):
                # 处理 context (premise)
                segments_pre = precess_rst_tree.merge_strings(
                    input_sentences_batch[index * 2],
                    all_segmentation_pred_batch[index * 2],
                )

                # 处理 output (hypothesis)
                segments_hyp = precess_rst_tree.merge_strings(
                    input_sentences_batch[index * 2 + 1],
                    all_segmentation_pred_batch[index * 2 + 1],
                )

                # 处理 context 的 RST 结果
                if all_tree_parsing_pred_batch[index * 2][0] == "NONE":
                    node_number_pre = 1
                    node_string_pre = [segments_pre]
                    RelationAndNucleus_pre = "NONE"
                    tree_pre = [[1, 1]]
                    leaf_node_pre = [1]
                    parent_dict_pre = {"1_1": 1}
                    print(f"样本 {count}: context 无 RST 结构")
                else:
                    rst_info_pre = all_tree_parsing_pred_batch[index * 2][0].split()
                    node_number_pre, node_string_pre = precess_rst_tree.use_rst_info(
                        rst_info_pre, segments_pre
                    )
                    RelationAndNucleus_pre = precess_rst_tree.get_RelationAndNucleus(
                        rst_info_pre
                    )
                    parent_dict_pre, leaf_node_pre, tree_pre = self.get_tree(
                        node_number_pre
                    )

                # 处理 output 的 RST 结果
                if all_tree_parsing_pred_batch[index * 2 + 1][0] == "NONE":
                    node_number_hyp = 1
                    node_string_hyp = [segments_hyp]
                    RelationAndNucleus_hyp = "NONE"
                    tree_hyp = [[1, 1]]
                    leaf_node_hyp = [1]
                    parent_dict_hyp = {"1_1": 1}
                    print(f"样本 {count}: output 无 RST 结构")
                else:
                    rst_info_hyp = all_tree_parsing_pred_batch[index * 2 + 1][0].split()
                    node_number_hyp, node_string_hyp = precess_rst_tree.use_rst_info(
                        rst_info_hyp, segments_hyp
                    )
                    RelationAndNucleus_hyp = precess_rst_tree.get_RelationAndNucleus(
                        rst_info_hyp
                    )
                    parent_dict_hyp, leaf_node_hyp, tree_hyp = self.get_tree(
                        node_number_hyp
                    )

                rst_results.append(
                    {
                        "pre_node_number": node_number_pre,
                        "pre_node_string": node_string_pre,
                        "pre_node_relations": RelationAndNucleus_pre,
                        "pre_tree": tree_pre,
                        "pre_leaf_node": leaf_node_pre,
                        "pre_parent_dict": parent_dict_pre,
                        "hyp_node_number": node_number_hyp,
                        "hyp_node_string": node_string_hyp,
                        "hyp_node_relations": RelationAndNucleus_hyp,
                        "hyp_tree": tree_hyp,
                        "hyp_leaf_node": leaf_node_hyp,
                        "hyp_parent_dict": parent_dict_hyp,
                    }
                )

                if count % 100 == 0:
                    print(f"已处理: {count} 个样本")
                count += 1

                # 每1000个样本保存一次
                if count % 1000 == 0:
                    os.makedirs(self.save_dir, exist_ok=True)
                    rst_name = f"{count}_rst_result.jsonl"
                    self.write_jsonl(os.path.join(self.save_dir, rst_name), rst_results)
                    rst_results = []

        # 保存剩余结果
        if rst_results and count < 1000:
            os.makedirs(self.save_dir, exist_ok=True)
            self.write_jsonl(rst_results_store_path, rst_results)
            print(f"保存了 {len(rst_results)} 条 RST 结果")
        elif rst_results:
            os.makedirs(self.save_dir, exist_ok=True)
            rst_name = f"{count}_left_rst_result.jsonl"
            self.write_jsonl(os.path.join(self.save_dir, rst_name), rst_results)

        print(f"剩余 RST 结果长度: {len(rst_results)}")
        return rst_results

    def get_stored_rst(self, path):
        """读取已保存的 RST 结果"""
        rst_results = []
        with open(path, "r") as file:
            for line in file:
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
        print(f"从 {path} 读取了已保存的 RST 结果")
        return rst_results


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class ModernBERTEmbedder:
    """
    使用 ModernBERT 生成 embeddings

    改编自 RSTEmbedder，使用 ModernBERT 替代 XLM-RoBERTa
    """

    def __init__(self, model_path, save_dir, purpose, save_or_not):
        """
        Args:
            model_path: ModernBERT 模型路径
            save_dir: 保存根目录
            purpose: 用途 (train/test)
            save_or_not: 是否保存
        """
        print(f"正在加载 ModernBERT 模型: {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # 使用 float16 减少显存占用

            self.model = AutoModel.from_pretrained(
                model_path,
                # torch_dtype=torch.float16,  # 使用半精度
                device_map="auto"
            )
            print("✅ ModernBERT 模型加载成功")
        except Exception as e:
            print(f"❌ ModernBERT 模型加载失败: {e}")
            print("尝试使用标准加载方式...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            print("✅ ModernBERT 模型加载成功 (标准模式)")

        self.save_dir_lexical = os.path.join(save_dir, purpose)
        ensure_parent_directory(self.save_dir_lexical)
        self.save_or_not = save_or_not
        self.lexical_matrix_path = "lexical_matrixes.pkl"

    def write_jsonl(self, path, data):
        """写入 JSONL 文件"""
        with open(path, "w", encoding="utf-8") as file:
            for record in data:
                json_record = json.dumps(record, ensure_ascii=False)
                file.write(json_record + "\n")
        print(f"成功保存到 {path}")

    def get_stored_rst(self, paths):
        """读取保存的 RST 结果"""
        rst_results = []
        if isinstance(paths, list):
            for path in paths:
                with open(path, "r") as file:
                    for line in file:
                        rst_dict = json.loads(line.strip())
                        rst_results.append(rst_dict)
        elif isinstance(paths, str):
            with open(paths, "r") as file:
                for line in file:
                    rst_dict = json.loads(line.strip())
                    rst_results.append(rst_dict)
        print("已读取保存的 RST 结果")
        return rst_results

    @staticmethod
    def find_leaf_node(number_list, all_string):
        """
        查找叶子节点的字符串及其对应的节点表示

        Args:
            number_list: 节点编号列表
            all_string: 所有节点的字符串列表

        Returns:
            leaf_string: 叶子节点字符串列表
            leaf_node_index: 叶子节点索引列表
        """
        leaf_node_index = []
        leaf_string = []

        for index, sub_list in enumerate(number_list):
            if sub_list[0] == sub_list[1]:
                leaf_string.append(all_string[index][0])
                leaf_node_index.append(index * 2 + 1)
            if sub_list[2] == sub_list[3]:
                leaf_string.append(all_string[index][1])
                leaf_node_index.append(index * 2 + 2)

        if len(leaf_string) == 0:
            raise Exception("未找到叶子节点！")

        return leaf_string, leaf_node_index

    def get_modernbert_embeddings_in_batches(self, texts, batch_size):
        """
        使用 ModernBERT 批量生成 embeddings

        Args:
            texts: 文本列表
            batch_size: 批次大小

        Returns:
            embeddings: embedding 列表
        """
        embeddings = []

        self.model.to(device)
        self.model.eval()

        dataset = TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch_idx, batch_texts in enumerate(tqdm(dataloader, desc="生成 embeddings")):
            try:
                # 清理GPU缓存
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}

                with torch.no_grad():
                    try:
                        outputs = self.model(**inputs, output_hidden_states=True)

                        if outputs is not None and outputs.last_hidden_state is not None:
                            # 获取最后一层隐藏状态的平均值
                            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                            embeddings.extend(batch_embeddings)
                        else:
                            print(f"警告: batch {batch_idx} 输出为 None")
                            # 创建零向量作为备用
                            batch_embeddings = np.zeros((len(batch_texts), 1024))
                            embeddings.extend(batch_embeddings)

                    except RuntimeError as e:
                        print(f"\n错误: batch {batch_idx} 处理失败: {e}")
                        print(f"Batch 大小: {len(batch_texts)}")
                        # 清理GPU并重试小批次
                        torch.cuda.empty_cache()

                        # 单个处理这个batch
                        for single_text in batch_texts:
                            try:
                                single_input = self.tokenizer(
                                    [single_text],
                                    return_tensors="pt",
                                    truncation=True,
                                    padding=True,
                                    max_length=512
                                )
                                single_input = {key: value.to(device) for key, value in single_input.items()}

                                single_output = self.model(**single_input, output_hidden_states=True)
                                single_emb = single_output.last_hidden_state.mean(dim=1).cpu().numpy()
                                embeddings.extend(single_emb)
                            except Exception as single_e:
                                print(f"单个文本处理失败，使用零向量: {single_e}")
                                embeddings.append(np.zeros(1024))

            except Exception as e:
                print(f"分词错误 batch {batch_idx}: {e}")
                # 使用零向量代替
                batch_embeddings = np.zeros((len(batch_texts), 1024))
                embeddings.extend(batch_embeddings)

        return embeddings

    def load_json(self, json_path):
        """加载 JSON 文件"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get_node_string_pair(self, rst_results, output_file="node_embeddings.npz"):
        """
        获取每个节点的字符串表示及其对应的 embeddings

        Args:
            rst_results: RST 结果列表
            output_file: 输出文件路径

        Returns:
            data_to_save: 保存的数据
        """
        ensure_parent_directory(output_file)
        directory = os.path.dirname(output_file)

        # 检查是否已存在
        npz_files = glob.glob(os.path.join(directory, "*.npz"))
        if npz_files:
            print(f"找到已有的 .npz 文件: {npz_files}")
            return

        print(f"新 RST 结果长度: {len(rst_results)}")
        data_to_save = []

        premise_texts = []
        hypothesis_texts = []
        premise_indices = []
        hypothesis_indices = []

        # 收集所有叶子节点文本
        for index, rst_result in enumerate(rst_results):
            if index % 100 == 0:
                print(f"正在处理索引: {index}")

            pre_leaf_node_string_list = rst_result["leaf_node_string_pre"]
            pre_leaf_node_index, pre_leaf_string = zip(*pre_leaf_node_string_list)
            premise_texts.extend(pre_leaf_string)
            premise_indices.append(pre_leaf_node_index)

            hyp_leaf_node_string_list = rst_result["leaf_node_string_hyp"]
            hyp_leaf_node_index, hyp_leaf_string = zip(*hyp_leaf_node_string_list)
            hypothesis_texts.extend(hyp_leaf_string)
            hypothesis_indices.append(hyp_leaf_node_index)

        # 批量生成 embeddings
        print("正在为 premise 生成 embeddings...")
        premise_embeddings = self.get_modernbert_embeddings_in_batches(
            premise_texts, batch_size=32  # 减小批次大小避免CUDA错误
        )

        print("正在为 hypothesis 生成 embeddings...")
        hypothesis_embeddings = self.get_modernbert_embeddings_in_batches(
            hypothesis_texts, batch_size=32  # 减小批次大小避免CUDA错误
        )

        # 重新组织 embedding 结果
        premise_offset = 0
        hypothesis_offset = 0

        for i, rst_result in enumerate(rst_results):
            if i % 100 == 0:
                print(f"正在保存索引 {i} 的数据")

            node_embeddings_premise = [
                (
                    node,
                    premise_embeddings[premise_offset + j],
                    premise_texts[premise_offset + j],
                )
                for j, node in enumerate(premise_indices[i])
            ]
            premise_offset += len(premise_indices[i])

            node_embeddings_hypothesis = [
                (
                    node,
                    hypothesis_embeddings[hypothesis_offset + j],
                    hypothesis_texts[hypothesis_offset + j],
                )
                for j, node in enumerate(hypothesis_indices[i])
            ]
            hypothesis_offset += len(hypothesis_indices[i])

            data_to_save.append(
                {
                    "premise": node_embeddings_premise,
                    "hypothesis": node_embeddings_hypothesis,
                }
            )

            # 每1000个样本保存一次
            if (i % 1000) == 0 and (i != 0):
                filename = output_file.replace('.npz', f'_{i}.npz')
                torch.save(data_to_save, filename)
                data_to_save = []
                print(f"已保存 {i} 对")

        # 保存剩余数据
        if data_to_save and i < 1000:
            filename = output_file
            torch.save(data_to_save, filename)
        elif data_to_save:
            filename = output_file.replace('.npz', f'_{i}.npz')
            torch.save(data_to_save, filename)

        print("所有 embeddings 已生成")
        return data_to_save

    def load_embeddings(self, file_path):
        """加载 embeddings"""
        data = torch.load(file_path)
        return data

    def rewrite_rst_result(self, rst_results_store_paths, new_rst_results_store_path):
        """
        重写 RST 结果，提取每个节点的核性和关系，便于构建 DGL 图

        Args:
            rst_results_store_paths: 原始 RST 结果存储路径
            new_rst_results_store_path: 新 RST 结果存储路径

        Returns:
            new_rst_results: 重写后的 RST 结果
        """
        ensure_parent_directory(new_rst_results_store_path)

        if os.path.exists(new_rst_results_store_path):
            print("新 RST 结果已存在")
            new_rst_results = self.get_stored_rst(new_rst_results_store_path)
            return new_rst_results

        rst_results = self.get_stored_rst(rst_results_store_paths)
        new_rst_results = []

        for rst_result in tqdm(rst_results, desc="重写 RST 结果"):
            single_dict = {}
            rst_relation_premise = []
            rst_relation_hypothesis = []
            premise_node_nuclearity = [(0, "root")]
            hypothesis_node_nuclearity = [(0, "root")]

            # 处理 premise (context)
            if rst_result["pre_node_number"] == 1:
                premise_node_nuclearity.append((1, "single"))
                single_dict["premise_node_nuclearity"] = premise_node_nuclearity
                single_dict["rst_relation_premise"] = ["NONE"]
                single_dict["pre_node_type"] = [1, 0]
                single_dict["leaf_node_string_pre"] = [
                    [1, rst_result["pre_node_string"][0]]
                ]
            else:
                pre_leaf_string, pre_leaf_node_index = self.find_leaf_node(
                    rst_result["pre_node_number"], rst_result["pre_node_string"]
                )

                if len(pre_leaf_string) != len(pre_leaf_node_index):
                    raise ValueError("叶子节点数量不匹配")

                combined_list_pre = list(zip(pre_leaf_node_index, pre_leaf_string))
                single_dict["leaf_node_string_pre"] = combined_list_pre

                pre_rel = rst_result["pre_node_relations"]
                pre_tree = rst_result["pre_tree"]

                for index, item in enumerate(pre_rel):
                    rel_left = item["rel_left"]
                    src_left = pre_tree[index * 2][0] - 1
                    dst_left = pre_tree[index * 2][1] - 1
                    node_nuclearity = item["nuc_left"]
                    relation_1 = (src_left, dst_left, rel_left)
                    node_nuclearity_1 = (dst_left, node_nuclearity)
                    rst_relation_premise.append(relation_1)
                    premise_node_nuclearity.append(node_nuclearity_1)

                    rst_right = item["rel_right"]
                    src_right = pre_tree[index * 2 + 1][0] - 1
                    dst_right = pre_tree[index * 2 + 1][1] - 1
                    node_nuclearity = item["nuc_right"]
                    relation_2 = (src_right, dst_right, rst_right)
                    node_nuclearity_2 = (dst_right, node_nuclearity)
                    rst_relation_premise.append(relation_2)
                    premise_node_nuclearity.append(node_nuclearity_2)

                pre_child_node_list = [x - 1 for x in rst_result["pre_leaf_node"]]
                pre_node_type = [
                    0 if i in pre_child_node_list else 1
                    for i in range(len(pre_tree) + 1)
                ]

                single_dict["rst_relation_premise"] = rst_relation_premise
                single_dict["premise_node_nuclearity"] = premise_node_nuclearity
                single_dict["pre_node_type"] = pre_node_type

            # 处理 hypothesis (output)
            if rst_result["hyp_node_number"] == 1:
                hypothesis_node_nuclearity.append((1, "single"))
                single_dict["hypothesis_node_nuclearity"] = hypothesis_node_nuclearity
                single_dict["rst_relation_hypothesis"] = ["NONE"]
                single_dict["hyp_node_type"] = [1, 0]
                single_dict["leaf_node_string_hyp"] = [
                    [1, rst_result["hyp_node_string"][0]]
                ]
            else:
                hyp_leaf_string, hyp_leaf_node_index = self.find_leaf_node(
                    rst_result["hyp_node_number"], rst_result["hyp_node_string"]
                )

                if len(hyp_leaf_string) != len(hyp_leaf_node_index):
                    raise ValueError("叶子节点数量不匹配")

                combined_list_hyp = list(zip(hyp_leaf_node_index, hyp_leaf_string))
                single_dict["leaf_node_string_hyp"] = combined_list_hyp

                hyp_rel = rst_result["hyp_node_relations"]
                hyp_tree = rst_result["hyp_tree"]

                for index, item in enumerate(hyp_rel):
                    rel_left = item["rel_left"]
                    src_left = hyp_tree[index * 2][0] - 1
                    dst_left = hyp_tree[index * 2][1] - 1
                    node_nuclearity = item["nuc_left"]
                    relation_1 = (src_left, dst_left, rel_left)
                    node_nuclearity_1 = (dst_left, node_nuclearity)
                    rst_relation_hypothesis.append(relation_1)
                    hypothesis_node_nuclearity.append(node_nuclearity_1)

                    rst_right = item["rel_right"]
                    src_right = hyp_tree[index * 2 + 1][0] - 1
                    dst_right = hyp_tree[index * 2 + 1][1] - 1
                    node_nuclearity = item["nuc_right"]
                    relation_2 = (src_right, dst_right, rst_right)
                    node_nuclearity_2 = (dst_right, node_nuclearity)
                    rst_relation_hypothesis.append(relation_2)
                    hypothesis_node_nuclearity.append(node_nuclearity_2)

                hyp_child_node_list = [x - 1 for x in rst_result["hyp_leaf_node"]]
                hyp_node_type = [
                    0 if i in hyp_child_node_list else 1
                    for i in range(len(hyp_tree) + 1)
                ]

                single_dict["rst_relation_hypothesis"] = rst_relation_hypothesis
                single_dict["hypothesis_node_nuclearity"] = hypothesis_node_nuclearity
                single_dict["hyp_node_type"] = hyp_node_type

            new_rst_results.append(single_dict)

        self.write_jsonl(new_rst_results_store_path, new_rst_results)
        return new_rst_results

    def find_lexical_chains(
        self, rst_results, node_features1, node_features2, threshold=0.8
    ):
        """
        查找两个文本之间的词汇链

        Args:
            rst_results: RST 结果
            node_features1: 节点特征 1
            node_features2: 节点特征 2
            threshold: 相似度阈值

        Returns:
            chains_matrix: 词汇链矩阵
        """
        pre_length = len(rst_results["pre_node_type"])
        pre2_length = len(rst_results["hyp_node_type"])

        chains_matrix = np.zeros((pre_length, pre2_length))

        for node_id1, embedding1, _ in node_features1:
            emb1 = np.array(embedding1)
            if emb1.ndim == 1:
                emb1 = emb1.reshape(1, -1)

            for node_id2, embedding2, _ in node_features2:
                emb2 = np.array(embedding2)
                if emb2.ndim == 1:
                    emb2 = emb2.reshape(1, -1)

                similarity = cosine_similarity(emb1, emb2)[0][0]

                if similarity > threshold:
                    chains_matrix[node_id1][node_id2] = 1

        # 归一化
        amin, amax = chains_matrix.min(), chains_matrix.max()
        epsilon = 1e-7
        chains_matrix = (chains_matrix - amin) / (amax - amin + epsilon)

        return chains_matrix

    def save_lexical_matrix(self, path, matrixes):
        """保存词汇链矩阵"""
        with open(path, "wb") as f:
            pickle.dump(matrixes, f)

    def load_lexical_matrix(self, filename):
        """加载词汇链矩阵"""
        with open(filename, "rb") as f:
            matrixes = pickle.load(f)
        return matrixes

    def store_or_get_lexical_matrixes(
        self, train_re_rst_result_path, emb_path, lexical_matrixes_path
    ):
        """
        保存或获取词汇链矩阵

        Args:
            train_re_rst_result_path: RST 结果路径
            emb_path: embedding 路径
            lexical_matrixes_path: 词汇链矩阵路径

        Returns:
            matrixes: 词汇链矩阵列表
        """
        ensure_parent_directory(lexical_matrixes_path)
        rst_results = self.get_stored_rst(train_re_rst_result_path)

        if os.path.exists(lexical_matrixes_path):
            matrixes = self.load_lexical_matrix(lexical_matrixes_path)
            print(f"矩阵形状: {matrixes[0].shape}")
            non_zero_indices = np.nonzero(matrixes[0])
            print(f"非零索引: {non_zero_indices}")
            print("已加载保存的词汇链矩阵")
            return matrixes

        embeddings = self.load_embeddings(emb_path)
        matrixes = []

        for index, rst_result in enumerate(tqdm(rst_results, desc="计算词汇链")):
            node_features1 = extract_node_features(embeddings, index, "premise")
            node_features2 = extract_node_features(embeddings, index, "hypothesis")
            matrix = self.find_lexical_chains(
                rst_result, node_features1, node_features2
            )
            matrixes.append(matrix)

        if self.save_or_not:
            os.makedirs(self.save_dir_lexical, exist_ok=True)
            print("保存词汇链矩阵")
            self.save_lexical_matrix(lexical_matrixes_path, matrixes)

        return matrixes


def load_all_data(data_processor, data_path, rst_path):
    """
    加载所有数据

    Args:
        data_processor: 数据处理器
        data_path: 数据路径
        rst_path: RST 结果路径

    Returns:
        data: 数据
        rst_results: RST 结果
    """
    data = data_processor.load_json(data_path)
    rst_results = data_processor.get_rst(data, rst_path)
    return data, rst_results


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large"
    OVERALL_SAVE_DIR = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data"
    GRAPH_INFOS_DIR = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_info"

    ensure_parent_directory(OVERALL_SAVE_DIR)

    # 训练集路径
    TRAIN_DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/hallucination_train.json"
    TRAIN_RST_RESULT_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/train/rst_result.jsonl"
    TRAIN_RE_RST_RESULT_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/train/new_rst_result.jsonl"
    TRAIN_PRE_EMB_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/train/node_embeddings.npz"
    TRAIN_LEXICAL_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_info/train/lexical_matrixes.pkl"

    # 测试集路径
    TEST_DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/hallucination_test.json"
    TEST_RST_RESULT_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/test/rst_result.jsonl"
    TEST_RE_RST_RESULT_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/test/new_rst_result.jsonl"
    TEST_PRE_EMB_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/test/node_embeddings.npz"
    TEST_LEXICAL_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_info/test/lexical_matrixes.pkl"

    print("="*60)
    print("开始处理训练集")
    print("="*60)

    # # 处理训练集
    # data_processor_train = HallucinationDataProcessor(True, OVERALL_SAVE_DIR, "train")
    # train_data, train_rst_result = load_all_data(
    #     data_processor_train, TRAIN_DATA_PATH, TRAIN_RE_RST_RESULT_PATH
    # )

    # embedder_train = ModernBERTEmbedder(MODEL_PATH, GRAPH_INFOS_DIR, "train", True)

    # train_rst_results_store_paths = glob.glob(
    #     os.path.join(os.path.join(OVERALL_SAVE_DIR, "train"), "*.jsonl")
    # )
    # print(f"训练集 RST 结果路径: {train_rst_results_store_paths}")

    # train_new_rst_results = embedder_train.rewrite_rst_result(
    #     train_rst_results_store_paths,
    #     TRAIN_RE_RST_RESULT_PATH,
    # )

    # # train_node_string_pairs = embedder_train.get_node_string_pair(
    # #     train_new_rst_results, TRAIN_PRE_EMB_PATH
    # # )

    # train_matrix = embedder_train.store_or_get_lexical_matrixes(
    #     TRAIN_RE_RST_RESULT_PATH, TRAIN_PRE_EMB_PATH, TRAIN_LEXICAL_PATH
    # )

    print("\n" + "="*60)
    print("开始处理测试集")
    print("="*60)

    # 处理测试集
    data_processor_test = HallucinationDataProcessor(True, OVERALL_SAVE_DIR, "test")
    test_data, test_rst_result = load_all_data(
        data_processor_test, TEST_DATA_PATH, TEST_RST_RESULT_PATH
    )

    embedder_test = ModernBERTEmbedder(MODEL_PATH, GRAPH_INFOS_DIR, "test", True)

    test_rst_results_store_paths = glob.glob(
        os.path.join(os.path.join(OVERALL_SAVE_DIR, "test"), "*.jsonl")
    )
    print(f"测试集 RST 结果路径: {test_rst_results_store_paths}")

    test_new_rst_results = embedder_test.rewrite_rst_result(
        test_rst_results_store_paths,
        TEST_RE_RST_RESULT_PATH,
    )

    test_node_string_pairs = embedder_test.get_node_string_pair(
        test_new_rst_results, TEST_PRE_EMB_PATH
    )

    test_matrix = embedder_test.store_or_get_lexical_matrixes(
        TEST_RE_RST_RESULT_PATH, TEST_PRE_EMB_PATH, TEST_LEXICAL_PATH
    )

    print("\n" + "="*60)
    print("数据处理完成!")
    print("="*60)
