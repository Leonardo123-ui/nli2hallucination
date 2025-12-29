from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from data_loader import RSTDataset
import logging
import torch

# setlog
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)


@dataclass
class DataPaths:
    rst_path: str
    nli_data_path: str
    emb_path: str
    lexical_path: str
    pair_graph: str


class Config:
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 42)
        self.stage = kwargs.get("stage", "classification")
        self.mode = kwargs.get("mode", "train")
        # basic path
        self.base_dir = kwargs.get("base_dir", "/mnt/nlp/yuanmengying")
        self.batch_file_size = kwargs.get("batch_file_size", 1)

        # training set
        self.epochs = kwargs.get("epochs", 7)
        self.batch_size = kwargs.get("batch_size", 7)
        self.save_dir = kwargs.get("save_dir", "checkpoints")
        self.save_interval = kwargs.get("save_interval", 5)
        self.log_interval = kwargs.get("log_interval", 100)
        self.use_tensorboard = kwargs.get("use_tensorboard", True)
        self.tensorboard_dir = kwargs.get("tensorboard_dir", "runs")
        self.eval_interval = kwargs.get("eval_interval", 5)
        self.train_file_per_epoch = kwargs.get("train_file_per_epoch", 2)
        
        # optimizer setting
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)
        self.lr = kwargs.get("lr", 0.001)
        self.total_steps = kwargs.get("total_steps", 1000)
        self.optimizer_type = kwargs.get("optimizer_type", "adamw")
        self.scheduler_type = kwargs.get("scheduler_type", "linear_warmup")
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # model setting
        self.model_config = kwargs.get(
            "model_config",
            {
                "in_dim": 1024,
                "hidden_dim": 1024,
                "n_classes": 2,
                "rel_names": [
                    "Temporal",
                    "TextualOrganization",
                    "Joint",
                    "Topic-Comment",
                    "Comparison",
                    "Condition",
                    "Contrast",
                    "Evaluation",
                    "Topic-Change",
                    "Summary",
                    "Manner-Means",
                    "Attribution",
                    "Cause",
                    "Background",
                    "Enablement",
                    "Explanation",
                    "Same-Unit",
                    "Elaboration",
                    "span",
                    "lexical",
                ],
            },
        )

        # init path
        self._init_data_paths()

    def _init_data_paths(self):
        """init all data path"""
        self.paths = {
            "train": DataPaths(
                rst_path=(
                    r"/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/train/new_rst_result.jsonl"
                ),
                nli_data_path=(
                    r"/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/hallucination_train.json"
                ),
                emb_path=(
                    "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/train/node_embeddings.npz"
                ),
                lexical_path="/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_info/train/lexical_matrixes.pkl",
                pair_graph=r"/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_pairs/train",
                # pair_graph=r"/mlx/users/mengying.yuan/nli_code/data_processed/graph_pairs_shuffle_v3/train",
            ),
            "test": DataPaths(
                rst_path=(
                    r"/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/test/new_rst_result.jsonl"
                ),
                nli_data_path=(
                    r"/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/hallucination_test.json"
                ),
                emb_path=(
                    "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/test/node_embeddings.npz"
                ),
        
                lexical_path="/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_info/test/lexical_matrixes.pkl",
                
                pair_graph=r"/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_pairs/test",  
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """set instance from dict"""
        return cls(**config_dict)

    def to_dict(self):
        d = self.__dict__.copy()
        if "device" in d:
            d["device"] = str(d["device"])
        return d

    def get(self, key, default=None):
        return getattr(self, key, default)


def data_model_loader(device):
    # build config
    config = Config(
        save_dir="checkpoints/experiment1",
        tensorboard_dir="runs/experiment1",
        optimizer_type="adamw",
        scheduler_type="linear_warmup",
        device=device,
    )

    # create dataset
    logging.info("Processing train data")
    train_dataset = RSTDataset(
        config.paths["train"].rst_path,
        config.paths["train"].nli_data_path,
        config.paths["train"].emb_path,
        config.paths["train"].lexical_path,
        config.batch_file_size,
        save_dir=config.paths["train"].pair_graph,
    )
    logging.info("Processing test data")
    test_dataset = RSTDataset(
        config.paths["test"].rst_path,
        config.paths["test"].nli_data_path,
        config.paths["test"].emb_path,
        config.paths["test"].lexical_path,
        config.batch_file_size,
        save_dir=config.paths["test"].pair_graph,
    )
    config.total_steps = 4800 * config.epochs  # data_loader'length * epochs
    config.warmup_steps = int(config.total_steps * config.warmup_ratio)
    config.steps_per_epoch = 4800
    config.train_file_per_epoch = 1
    # init model
    return config, train_dataset, test_dataset


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, train_dataset, test_dataset = data_model_loader(device)
