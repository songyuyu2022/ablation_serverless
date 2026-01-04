# dataset.py
from __future__ import annotations
import os, json, random
from dataclasses import dataclass
from typing import Dict, Tuple
import torch

DATA_PATH_DEFAULT = "input.txt"
VOCAB_PATH_DEFAULT = "vocab.json"

def build_char_vocab(txt_path: str = DATA_PATH_DEFAULT, vocab_path: str = VOCAB_PATH_DEFAULT) -> Tuple[Dict[str,int], Dict[int,str]]:
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {ch:i for i,ch in enumerate(chars)}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(stoi, f, ensure_ascii=False, indent=2)
    itos = {int(i):ch for ch,i in stoi.items()}
    return stoi, itos

def load_text_as_ids(txt_path: str, stoi: Dict[str,int]) -> torch.Tensor:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = [stoi.get(ch, 0) for ch in text]
    return torch.tensor(ids, dtype=torch.long)

@dataclass
class DatasetSplit:
    data: torch.Tensor
    vocab_size: int

def make_splits(ids: torch.Tensor, vocab_size: int, train_frac: float = 0.9):
    n = int(ids.numel())
    n_train = max(1, int(n * train_frac))
    n_train = min(n - 1, n_train)
    return DatasetSplit(ids[:n_train].contiguous(), vocab_size), DatasetSplit(ids[n_train:].contiguous(), vocab_size)

class LMTextBatcher:
    def __init__(self, split: DatasetSplit, *, batch_size: int, seq_len: int, seed: int = 42):
        self.split = split
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.rng = random.Random(seed)
        if self.split.data.numel() < self.seq_len + 1:
            raise ValueError("Text too short")

    def next_batch(self, device: str = "cpu"):
        data = self.split.data
        max_start = int(data.numel() - (self.seq_len + 1))
        starts = [self.rng.randint(0, max_start) for _ in range(self.batch_size)]
        x = torch.stack([data[s:s+self.seq_len] for s in starts], dim=0)
        y = torch.stack([data[s+1:s+1+self.seq_len] for s in starts], dim=0)
        return x.to(device), y.to(device)
