import json
import torch
import pickle
import dgl
import os
import re
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from dgl.dataloading import GraphDataLoader
from build_base_graph_extract import build_graph
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_graph_pairs(graph_pairs, file_path):
    graphs = [graph for pair in graph_pairs for graph in pair]
    dgl.save_graphs(file_path, graphs)
    logging.info("Saved pair graph at %s", file_path)


def load_graph_pairs(file_path, num_pairs):
    graphs, _ = dgl.load_graphs(file_path)
    graph_pairs = [(graphs[i * 2], graphs[i * 2 + 1]) for i in range(num_pairs)]
    logging.info("Loaded from %s", file_path)
    return graph_pairs



def extract_node_features(embeddings_data, idx, prefix):
    node_features = []
    for item in embeddings_data[idx][prefix]:
        node_id, embedding, text = item
        node_features.append((node_id, embedding, text))
    return node_features


def load_all_embeddings(directory_path):
    embeddings_list = []

    # Get the list of filenames
    filenames = os.listdir(directory_path)

    # Define a function to extract numbers from filenames
    def extract_number(filename):
        # Use regular expressions to extract the numeric part of the filename
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else float("inf")

    # Sort filenames based on the extracted numbers
    sorted_filenames = sorted(filenames, key=extract_number)

    for filename in sorted_filenames:
        if filename.endswith(".npz"):
            file_path = os.path.join(directory_path, filename)
            embeddings = torch.load(file_path)
            embeddings_list.extend(embeddings)
            logging.info("Loaded embeddings from %s", file_path)

    return embeddings_list


def load_embeddings_from_directory(directory_path, limit=None):
    embeddings_list = []
    count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(directory_path, filename)
            embeddings = torch.load(file_path)
            embeddings_list.extend(embeddings)
            count += 1
            if limit and count >= limit:
                break
    return embeddings_list


class RSTDataset(Dataset):
    def __init__(
        self,
        rst_path,
        nli_data_path,  # Assumed data
        embeddings_path,
        lexical_matrix_path,
        batch_file_size=1,  # Number of files processed per batch
        save_dir="./graph_pairs",  # Directory to save graph_pairs
    ):
        self.rst_path = rst_path
        self.nli_data_path = nli_data_path
        self.emb_path = embeddings_path
        self.lexical_matrix_path = lexical_matrix_path
        self.batch_file_size = batch_file_size
        self.save_dir = save_dir

        # Ensure the save directory exists
        # os.makedirs(self.save_dir, exist_ok=True)
        # Store pre-built graphs to avoid building them during training
        self.graph_pairs = []

    def load_nli_labels(self, nli_data):
        # nli_labels length is 2/3 of the NLI data size
        # 从json文件中读取每个item的label

        nli_labels = []
        for item in nli_data:
            nli_labels.append(item["label"])
        return nli_labels  # Neutral cases are not calculated

    def load_rst_data(self):
        """
        Load RST data, only needs to be done once.
        """
        rst_results = []
        with open(self.rst_path, "r") as file:
            for line in file:
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
        return rst_results

    def load_nli_data(self):
        """
        Load model output, only needs to be done once.
        """
        with open(self.nli_data_path, "r", encoding="utf-8") as file:
            nli_data = json.load(file)
        return nli_data

    def load_batch_files(self, batch_num):
        """
        Load corresponding lexical chain and embedding files based on batch number,
        and build graphs. If saved graph_pairs file exists, load it directly.
        """
        # Filename for saved graph_pairs
        save_file = os.path.join(self.save_dir, f"graph_pairs_batch_{batch_num}.pkl")

        # If the file exists, load it directly
        if os.path.isfile(save_file):
            with open(save_file, "rb") as f:
                self.graph_pairs = pickle.load(f)
            logging.info("Loaded graph pairs from %s", save_file)
            return

        # Otherwise, load files, build graphs, and save
        print("embedding dir exists")
        # Get all file paths and sort them
        self.data = self.load_rst_data()
        self.nli_data = self.load_nli_data()
        self.nli_labels = self.load_nli_labels(self.nli_data)  # [[[],[]],] or []

        batch_lexical_chains = []
        batch_embeddings = []

        # Load lexical chains
        with open(self.lexical_matrix_path, "rb") as f:
            batch_lexical_chains.extend(pickle.load(f))

        # Load embeddings
        batch_embeddings.extend(torch.load(self.emb_path))

        # Build graphs based on lexical chains and embeddings
        self.graph_pairs = self.build_graphs(
            batch_lexical_chains, batch_embeddings, self.nli_labels
        )

        # Save the built graph_pairs
        with open(save_file, "wb") as f:
            pickle.dump(self.graph_pairs, f)

    def build_graphs(self, lexical_chains, embeddings, nli_labels):
        """
        Build graphs based on loaded lexical chains and embeddings.
        """
        graph_pairs = []
        for idx in range(len(embeddings)):
            count = idx
            rst_result = self.data[idx]
            # Build graphs
            node_features_premise = extract_node_features(embeddings, count, "premise")
            rst_relations_premise = rst_result["rst_relation_premise"]
            node_types_premise = rst_result["pre_node_type"]
            g_premise = build_graph(
                node_features_premise, node_types_premise, rst_relations_premise
            )

            node_features_hypothesis = extract_node_features(
                embeddings, count, "hypothesis"
            )
            rst_relations_hypothesis = rst_result["rst_relation_hypothesis"]
            node_types_hypothesis = rst_result["hyp_node_type"]
            g_hypothesis = build_graph(
                node_features_hypothesis,
                node_types_hypothesis,
                rst_relations_hypothesis,
            )
            nli_label = nli_labels[count]

            graph_pairs.append(
                (   
                    g_premise,
                    g_hypothesis,
                    lexical_chains[count],
                    nli_label,
                )
            )

        return graph_pairs

    def __len__(self):
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        (g_premise, g_hypothesis, lexical_chain, nli_label) = self.graph_pairs[
            idx
        ]

        return (
            g_premise,
            g_hypothesis,
            lexical_chain,
            nli_label,
        )


if __name__ == "__main__":
    base_data_dir = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data"
    dir = "graph_info"
    type_list = ["train", "test"]
    batch_file_size = 1
    for type in type_list:
        print("*" * 30, type, "*" * 30)
        train_rst_path = f"{base_data_dir}/{type}/new_rst_result.jsonl"
        train_nli_data_path = (
            f"{base_data_dir}/hallucination_{type}.json"  # train_re_hyp.json
        )

        train_emb_path = f"{base_data_dir}/{type}/node_embeddings.npz"

        train_lexical_path = (
            f"{base_data_dir}/{dir}/{type}/lexical_matrixes.pkl"
        )
        train_pair_graph = f"{base_data_dir}/graph_pairs/{type}"
        train_dataset = RSTDataset(
            train_rst_path,
            train_nli_data_path,
            train_emb_path,
            train_lexical_path,
            batch_file_size,
            save_dir=train_pair_graph,
        )
        train_dataset.load_batch_files(0)
        for batch_data in train_dataset:
            print("safe")