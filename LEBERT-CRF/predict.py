import torch
from transformers import BertTokenizer,BertConfig
from torch.utils.data import DataLoader,Dataset

import os
from tqdm import tqdm
import numpy as np

from data_utils import get_corpus_matched_word_from_lexicon_tree,ItemVocabLabel,ItemVocabArray,load_word_embedding,pretrained_embedding_for_corpus
from utils import set_train_args,seed_everything,build_lexicon_tree_from_vocabs
from model import LEBertCRF

args = set_train_args()
seed_everything(args.seed)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lexicon_tree = build_lexicon_tree_from_vocabs([args.pretrain_embed_path], scan_nums=[args.max_scan_num])
embed_lexicon_tree = lexicon_tree

args.train_file = os.path.join(args.data_path, 'train.npz')
matched_words = get_corpus_matched_word_from_lexicon_tree(args.train_file, embed_lexicon_tree)
word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False)#初始化匹配词词表
label_vocab = ItemVocabLabel(files=[args.label_file], is_word=False)
word_embedding, embed_dim=load_word_embedding(args.pretrain_embed_path,args.max_scan_num)
match_word_embedding=pretrained_embedding_for_corpus(word_vocab,word_embedding,embed_dim)#加载匹配词词向量

# 初始化模型
tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)
config = BertConfig.from_pretrained(args.config_name)
config.num_labels = label_vocab.item_size
config.add_layer = 0
config.word_vocab_size = match_word_embedding.shape[0]
config.word_embed_dim = match_word_embedding.shape[1]
state = torch.load(args.output_path)
model = LEBertCRF.from_pretrained(args.pretrain_model_path, config=config).to(args.device)
model.load_state_dict(state['model_state'])
print('predict阶段加载模型完成')

#数据集
def read_testdata(path):
    test_sents=[]
    test_lens=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f.read().split('\n'):
            test_sents.append(list(line))
            test_lens.append(len(list(line)))
    return test_sents,test_lens

class TestDataset:
    def __init__(self,words,tokenizer,label_vocab,word_vocab,args,trie_tree):
        self.words=words
        self.tokenizer=tokenizer
        self.label_vocab=label_vocab
        self.word_vocab=word_vocab
        self.max_word_num=args.max_word_num
        self.trie_tree=trie_tree
        self.data=self.load_data(self.words)

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
    def load_data(self,words):
        features=[]
        cls_token_id = self.tokenizer.cls_token_id
        for text in tqdm(words):
            char_index2words = self.get_char2words(text)
            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text)
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
            text = ''.join(text)
            feature = {
                'text': text, 'input_ids': input_ids, 'word_ids': word_ids_list
            }
            features.append(feature)
        return features

class TestNERDataset(Dataset):
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

        batch_len = len(input_ids)
        max_len=max([len(s) for s in input_ids])
        batch_data = [0] * np.ones((batch_len, max_len))
        batch_seg = [1] * np.ones((batch_len, max_len))
        batch_attention_mask = [0] * np.ones((batch_len, max_len))
        for j in range(batch_len):
            cur_len=len(input_ids[j])
            batch_data[j][:cur_len]=input_ids[j]
            batch_seg[j][:cur_len] = 0
            batch_attention_mask[j][:cur_len] = 1

        batch_data=torch.tensor(batch_data,dtype=torch.long).to(self.device)
        batch_seg = torch.tensor(batch_seg, dtype=torch.long).to(self.device)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.device)

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
            'word_mask': batch_matched_word_mask
        }
        return data

test_sents,test_lens=read_testdata('./dataset/sample_per_line_preliminary_A.txt')
test_features=TestDataset(test_sents,tokenizer,label_vocab,word_vocab,args,embed_lexicon_tree).data
test_dataset = TestNERDataset(test_features,args)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=test_dataset.collate_fn)
id2label = {_id: _label for _label, _id in list(label_vocab.item2idx.items())}


pred_tags=[]
for batch_idx, data in enumerate(tqdm(test_dataloader)):
    input_ids = data['input_ids'].to(args.device)
    token_type_ids = data['token_type_ids'].to(args.device)
    attention_mask = data['attention_mask'].to(args.device)
    word_ids = data['word_ids'].to(args.device)
    word_mask = data['word_mask'].to(args.device)
    logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, labels=None)
    label_mask = attention_mask.gt(0)
    batch_output=model.crf.decode(logits,mask=label_mask)
    pred_tags.extend([id2label.get(idx) for idx in indices][1:] for indices in batch_output)
assert sum([len(x) for x in pred_tags])==sum(test_lens)

line_results=[]
for sent,labels in zip(test_sents,pred_tags):
    line_result=[]
    for word,lable in zip(sent,labels):
        line_result.append((word,lable))
    line_results.append(line_result)

with open('submit.txt','w',encoding='utf-8') as f:
    for i,line_result in enumerate(line_results):
        for word,tag in line_result:
            f.write(f'{word} {tag}\n')
        if i<len(line_results)-1:
            f.write('\n')
print('finish')







