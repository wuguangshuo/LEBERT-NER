from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np


class TaskDataset:
    def __init__(self,words,labels,tokenizer,label_vocab,word_vocab,args,trie_tree):
        self.words=words
        self.labels=labels
        self.tokenizer=tokenizer
        self.label_vocab=label_vocab
        self.word_vocab=word_vocab
        self.max_word_num=args.max_word_num
        self.trie_tree=trie_tree
        self.data=self.load_data(self.words,self.labels)

    def get_char2words(self, text):
        text_len = len(text)
        char_index2words = [[] for _ in range(text_len)]
        for idx in range(text_len):
            sub_sent = text[idx:idx + self.trie_tree.max_depth]  # speed using max depth
            words = self.trie_tree.enumerateMatch(sub_sent)  # 找到以text[idx]开头的所有单词
            for word in words:
                start_pos = idx
                end_pos = idx + len(word)
                for i in range(start_pos, end_pos):
                    char_index2words[i].append(word)
        return char_index2words
    def load_data(self,words,labels):
        features=[]
        cls_token_id = self.tokenizer.cls_token_id
        o_label_id = self.label_vocab.convert_token_to_id('O')

        for text,label in tqdm(zip(words,labels)):
            char_index2words = self.get_char2words(text)
            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text)
            label_ids = [o_label_id] + self.label_vocab.convert_tokens_to_ids(label)

            word_ids_list = []
            word_pad_id = self.word_vocab.convert_token_to_id('<pad>')
            for words in char_index2words:
                words = words[:self.max_word_num]
                word_ids = self.word_vocab.convert_tokens_to_ids(words)
                word_pad_num = self.max_word_num - len(words)
                word_ids = word_ids + [word_pad_id] * word_pad_num
                word_ids_list.append(word_ids)
            # 开头和结尾进行padding
            word_ids_list = [[word_pad_id] * self.max_word_num] + word_ids_list
            assert len(input_ids) == len(label_ids) == len(word_ids_list)
            text = ''.join(text)
            feature = {
                'text': text, 'input_ids': input_ids, 'word_ids': word_ids_list, 'label_ids': label_ids
            }
            features.append(feature)
        return features

class NERDataset(Dataset):
    def __init__(self, features,args):
        self.features = features
        self.device=args.device
        self.max_word_num=args.max_word_num

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        return feature
    def collate_fn(self,feature):
        text=[x['text'] for x in feature]
        input_ids=[x['input_ids'] for x in feature]
        word_ids = [x['word_ids'] for x in feature]
        label_ids = [x['label_ids'] for x in feature]

        batch_len = len(input_ids)
        max_len=max([len(s) for s in input_ids])
        batch_data = [0] * np.ones((batch_len, max_len))
        batch_seg = [1] * np.ones((batch_len, max_len))
        batch_attention_mask = [0] * np.ones((batch_len, max_len))
        batch_labels = [-1] * np.ones((batch_len, max_len))
        for j in range(batch_len):
            cur_len=len(input_ids[j])
            batch_data[j][:cur_len]=input_ids[j]
            batch_seg[j][:cur_len] = 0
            batch_attention_mask[j][:cur_len] = 1
            batch_labels[j][:cur_len]=label_ids[j]

        batch_data=torch.tensor(batch_data,dtype=torch.long).to(self.device)
        batch_seg = torch.tensor(batch_seg, dtype=torch.long).to(self.device)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

        batch_matched_word_ids=[0] * np.ones((batch_len, max_len,self.max_word_num))
        batch_matched_word_ids=batch_matched_word_ids.tolist()
        for i, line in enumerate(word_ids):
            for j,tmp in enumerate(line):
                batch_matched_word_ids[i][j] = tmp
        batch_matched_word_ids = torch.tensor(batch_matched_word_ids, dtype=torch.long).to(self.device)
        batch_matched_word_mask = batch_matched_word_ids.gt(0).to(self.device)
        data={'text': text,
            'input_ids': batch_data,
            'attention_mask': batch_attention_mask,
            'token_type_ids': batch_seg,
            'word_ids': batch_matched_word_ids,
            'word_mask': batch_matched_word_mask,
            'label_ids': batch_labels,
        }
        return data



