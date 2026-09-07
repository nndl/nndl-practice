"""IMDB文本输入：与datasets/download.py生成的gzip文件配套。"""
from __future__ import annotations

import gzip
import random
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """复用下载脚本的小写、HTML换行与空白处理；保留附着在词上的标点。"""
    text = re.sub(r"\s+", " ", text.replace("<br />", " ")).strip().lower()
    return text.split() if text else []


def load_imdb_data(data_path):
    """返回训练/验证/测试文本及标签，分别为20k/5k/25k条。"""
    root = Path(data_path)

    def read_split(name):
        rows = []
        with gzip.open(root / f"{name}.txt.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    label, text = line.rstrip("\n").split("\t", 1)
                    rows.append((text, int(label)))
        return rows

    return read_split("train"), read_split("dev"), read_split("test")


def load_vocab(vocab_path):
    """加载训练集构建的词表；0为补齐符，1为未知词。"""
    with gzip.open(vocab_path, "rt", encoding="utf-8") as stream:
        words = [line.rstrip("\n") for line in stream]
    if words[:2] != ["[PAD]", "[UNK]"] or len(set(words)) != len(words):
        raise ValueError("vocabulary must contain unique words starting with [PAD], [UNK]")
    return {word: index for index, word in enumerate(words)}


def balanced_subset(examples, size, seed=100):
    """从二分类影评中固定抽取正、负各半的子集。"""
    if size <= 0 or size % 2:
        raise ValueError("size must be a positive even number")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        group = [row for row in examples if row[1] == label]
        selected.extend(rng.sample(group, size // 2))
    rng.shuffle(selected)
    return selected


def build_vocab(train_examples, max_words=50000):
    """仅使用传入的训练文本统计词频，预留补齐和未知词。"""
    counts = Counter(word for text, _ in train_examples for word in tokenize(text))
    vocabulary = {"[PAD]": 0, "[UNK]": 1}
    for word, _ in counts.most_common(max_words):
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    return vocabulary


# 第6章的可复用数据与基线组件，供第8章保持同一实验条件。
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IMDBDataset(Dataset):
    def __init__(self, examples, word2id_dict):
        super().__init__()
        # 词表，用于将词转为词表索引的数字
        self.word2id_dict =  word2id_dict
        # 加载后的数据集
        self.examples = self.words_to_id(examples)

    def words_to_id(self, examples):
        tmp_examples = []
        for idx, example in enumerate(examples):
            seq, label = example
            # 将词映射为词表索引的ID， 对于词表中没有的词用[UNK]对应的ID进行替代
            seq = [self.word2id_dict.get(word, self.word2id_dict['[UNK]']) for word in tokenize(seq)]
            if not seq:
                seq = [self.word2id_dict["[UNK]"]]
            label = int(label)
            tmp_examples.append([seq, label])
        return tmp_examples

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        return seq, label

    def __len__(self):
        return len(self.examples)

def imdb_collate_fn(batch_data, pad_val=0, max_seq_len=256):
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    seqs, seq_lens, labels = [], [], []
    max_len = 0
    for example in batch_data:
        seq, label = example
        # 对数据序列进行截断
        seq = seq[:max_seq_len]
        # 对数据截断并保存于seqs中
        seqs.append(seq)
        seq_lens.append(len(seq))
        labels.append(label)
        # 保存序列最大长度
        max_len = max(max_len, len(seq))
    # 对数据序列进行填充至最大长度
    for i in range(len(seqs)):
        seqs[i] = seqs[i] + [pad_val] * (max_len - len(seqs[i]))

    return torch.tensor(seqs), torch.tensor(seq_lens), torch.tensor(labels)

class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sequence_output, sequence_length):
        sequence_length = sequence_length.to(device=sequence_output.device).unsqueeze(-1)
        # 根据sequence_length生成掩码矩阵，用于对Padding位置的信息进行掩蔽
        max_len = sequence_output.shape[1]
        mask = torch.arange(max_len, device=sequence_output.device) < sequence_length
        mask = mask.to(sequence_output.dtype).unsqueeze(-1)
        # 对序列中Padding部分进行掩蔽
        sequence_output = torch.mul(sequence_output, mask)
        # 对序列中的向量取均值
        batch_mean_hidden = torch.div(torch.sum(sequence_output, dim=1), sequence_length.to(sequence_output.dtype))
        return batch_mean_hidden

class Model_BiLSTM_FC(nn.Module):
    def __init__(self, num_embeddings, input_size, hidden_size, num_classes=2):
        super().__init__()
        # 词表大小
        self.num_embeddings = num_embeddings
        # 词向量的维度
        self.input_size = input_size
        # LSTM隐藏单元数量
        self.hidden_size = hidden_size
        # 情感分类类别数量
        self.num_classes = num_classes
        # 实例化嵌入层
        self.embedding_layer = nn.Embedding(num_embeddings, input_size, padding_idx=0)
        # 实例化LSTM层，batch_first=True 表示输入形状是[batch, length, embedding]
        self.lstm_layer = nn.LSTM(input_size, hidden_size,
                                  batch_first=True, bidirectional=True)
        # 实例化汇聚层
        self.average_layer = AveragePooling()
        # 实例化线性层
        self.output_layer = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, sequence_length):
        # 获取词向量
        inputs_emb = self.embedding_layer(input_ids)
        # 打包变长序列，LSTM 内部会跳过 [PAD] 位置
        packed = pack_padded_sequence(inputs_emb, sequence_length.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm_layer(packed)
        # 解包回 [batch, length, 2*hidden]，[PAD] 处仍是 0
        sequence_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 使用汇聚层汇聚sequence_output
        batch_mean_hidden = self.average_layer(sequence_output, sequence_length)
        # 输出文本分类logits
        logits = self.output_layer(batch_mean_hidden)
        return logits
