import argparse
import random
import os
import numpy as np
from tqdm import tqdm
import collections
from sklearn.model_selection import train_test_split
import logging

import torch

def set_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='设置随机种子')
    parser.add_argument("--data_path", type=str, default="./dataset/", help='数据集存放路径')
    parser.add_argument("--output_path", type=str, default='./save/model.pkl', help='模型与预处理数据的存放位置')
    parser.add_argument("--log_dir", type=str, default='./log/train.log', help='日志的存放位置')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs/', help='tensorboard的存放位置')
    parser.add_argument("--pretrain_embed_path", type=str, default='./dataset/final_vocab.txt', help='预训练词向量路径')
    parser.add_argument("--label_file", default="./dataset/labels.txt", type=str)
    parser.add_argument("--config_name", default="./pretrained_path/config.json", type=str,help="the config of define model")
    parser.add_argument("--max_word_num", default=5, type=int)
    parser.add_argument("--max_scan_num", default=2000000, type=int, help="The boundary of data files")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help='Bert的学习率')
    parser.add_argument("--crf_learning_rate", type=float, default=1e-3, help='crf的学习率')
    parser.add_argument("--classifier_learning_rate", type=float, default=1e-5, help='classifier的学习率')
    parser.add_argument("--eps", default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument("--pretrain_model_path", default='./pretrained_path/', type=str, help="the pretrained bert path")

    args = parser.parse_args()
    return args

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self, use_single=True):
        self.root = TrieNode()
        self.max_depth = 0
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    def insert(self, word):
        current = self.root
        deep = 0
        for letter in word:
            current = current.children[letter]
            deep += 1
        current.is_word = True
        if deep > self.max_depth:
            self.max_depth = deep

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def enumerateMatch(self, str, space=""):
        matched = []
        while len(str) > self.min_len:
            if self.search(str):
                matched.insert(0, space.join(str[:])) # 短的词总是在最前面
            del str[-1]
        if len(matched) > 1 and len(matched[0]) == 1: # filter single character word
            matched = matched[1:]
        return matched


def build_lexicon_tree_from_vocabs(vocab_files, scan_nums=None):
    # 1.获取词汇表
    vocabs = set()
    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)
            for idx in tqdm(range(total_line_num)):
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                if len(list(word.strip()))==1:
                    continue
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)
    return lexicon_tree

def val_split(dataset_dir):
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"].tolist()
    labels = data["labels"].tolist()
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=0.2, random_state=0)
    return x_train, x_dev, y_train, y_dev

def set_logger(log_path):
    logger = logging.getLogger()#用logging.getLogger(name)方法进行初始化
    logger.setLevel(logging.INFO)#设置级别

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)#地址
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
