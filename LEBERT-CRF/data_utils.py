from tqdm import tqdm
import json
import numpy as np


def sent_to_matched_words_set(sent, lexicon_tree, max_word_num=None):
    sent_length = len(sent)
    matched_words_set = set()
    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]
        words = lexicon_tree.enumerateMatch(sub_sent)

        _ = [matched_words_set.add(word) for word in words]
    matched_words_set = list(matched_words_set)
    matched_words_set = sorted(matched_words_set)
    return matched_words_set

def get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree):
    total_matched_words = set()
    data = np.load(files, allow_pickle=True)
    words = data["words"].tolist()
    for sent in words:
        sent_matched_words = sent_to_matched_words_set(sent, lexicon_tree)
        _ = [total_matched_words.add(word) for word in sent_matched_words]
    total_matched_words = list(total_matched_words)
    total_matched_words = sorted(total_matched_words)
    with open("matched_word.txt", "w", encoding="utf-8") as f:
        for word in total_matched_words:
            f.write("%s\n"%(word))
    return total_matched_words

class ItemVocabArray():
    def __init__(self, items_array, is_word=False, has_default=False):
        self.items_array = items_array
        self.item2idx = {}
        self.idx2item = []
        self.item_size = 0
        self.is_word = is_word
        if not has_default and self.is_word:
            self.item2idx['<pad>'] = self.item_size
            self.idx2item.append('<pad>')
            self.item_size += 1
            self.item2idx['<unk>'] = self.item_size
            self.idx2item.append('<unk>')
            self.item_size += 1
        self.init_vocab()

    def init_vocab(self):
        for item in self.items_array:
            self.item2idx[item] = self.item_size
            self.idx2item.append(item)
            self.item_size += 1

    def get_item_size(self):
        return self.item_size

    def convert_item_to_id(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        elif self.is_word:
            unk = "<unk>" + str(len(item))
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_items_to_ids(self, items):
        return [self.convert_item_to_id(item) for item in items]

    def convert_id_to_item(self, id):
        return self.idx2item[id]

    def convert_ids_to_items(self, ids):
        return [self.convert_id_to_item(id) for id in ids]

    def get_size(self):
        return self.size

    def convert_token_to_id(self, token):
        if token in self.item2idx:
            return self.item2idx[token]
        else:
            return self.item2idx['<unk>']

    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, idx):
        return self.idx2item[idx]

    def convert_ids_to_tokens(self, ids):
        return [self.convert_id_to_token(ids) for ids in ids]


class ItemVocabLabel():
    def __init__(self, files, is_word=False):
        self.files = files
        self.item2idx = {}
        self.idx2item = []
        self.item_size = 0
        self.is_word = is_word
        self.init_vocab()

    def init_vocab(self):
        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    items = line.split()
                    item = items[0].strip()
                    self.item2idx[item] = self.item_size
                    self.idx2item.append(item)
                    self.item_size += 1

    def get_item_size(self):
        return self.item_size

    def convert_token_to_id(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        elif self.is_word:
            unk = "<unk>" + str(len(item))
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_tokens_to_ids(self, items):
        return [self.convert_token_to_id(item) for item in items]

    def convert_id_to_item(self, id):
        return self.idx2item[id]

    def convert_ids_to_items(self, ids):
        return [self.convert_id_to_item(id) for id in ids]

def load_word_embedding(word_embed_path, max_scan_num):
    word_embed_dict = dict()
    with open(word_embed_path, 'r', encoding='utf8') as f:
        for idx, line in tqdm(enumerate(f)):
            # 只扫描前max_scan_num个词向量
            if idx > max_scan_num:
                break
            items = line.strip().split()
            if idx == 0:
                assert len(items) == 2
                num_embed, word_embed_dim = items
                num_embed, word_embed_dim = int(num_embed), int(word_embed_dim)
            else:
                assert len(items) == word_embed_dim + 1
                word = items[0]
                embedding = np.empty([1, word_embed_dim])
                embedding[:] = items[1:]
                word_embed_dict[word] = embedding
    return word_embed_dict,word_embed_dim

def pretrained_embedding_for_corpus(word_vocab,word_embedding,embed_dim):
    scale = np.sqrt(3.0 / embed_dim)
    pretrained_emb = np.empty([word_vocab.item_size, embed_dim])
    matched = 0
    not_matched = 0
    for idx, word in enumerate(word_vocab.idx2item):
        if word in word_embedding:
            pretrained_emb[idx, :] = word_embedding[word]
            matched += 1
        else:
            pretrained_emb[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_matched += 1
    return pretrained_emb