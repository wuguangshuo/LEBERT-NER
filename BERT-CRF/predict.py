from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch
import numpy as np
import config
import os
from transformers import BertModel,BertTokenizer
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import json



class NERDataset(Dataset):
    def __init__(self,words,config,label2id,word_pad_idx=0):
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
        self.label2id=label2id
        self.dataset=self.preprocess(words)
        self.word_pad_idx = word_pad_idx
        self.device = config.device
    def preprocess(self,origin_sentences):
        data=[]
        for line in origin_sentences:
            line=list(line)
            n=len(line)
            token_start_idxs=[]
            for i in range(n):
                token_start_idxs.extend([i+1])
            token_start_idxs=np.array(token_start_idxs)
            words=['[CLS]']+line
            data.append((self.tokenizer.convert_tokens_to_ids(words),token_start_idxs))
        return data
    def __getitem__(self, idx):
        word=self.dataset[idx]
        return word
    def __len__(self):
        return len(self.dataset)
    def collate_fn(self,batch):
        sentences=[x[0] for x in batch]
        batch_len = len(sentences)
        max_len=max([len(s) for s in sentences])
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts=[]
        # 解码时的label_mask
        batch_mask = self.word_pad_idx * np.ones((batch_len, max_len-1))
        for j in range(batch_len):
            cur_len=len(sentences[j])
            batch_data[j][:cur_len]=sentences[j]
        for j in range(batch_len):
            cur_len = len(sentences[j])-1
            batch_mask[j][:cur_len]=1
        #去除cls和pad对bilstm模型的影响
        for j in range(batch_len):
            label_start_idx = batch[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
        batch_data=torch.tensor(batch_data,dtype=torch.long).to(self.device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.uint8).to(self.device)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long).to(self.device)
        return [batch_data,batch_mask,batch_label_starts]
#读取数据
def read_testdata(path):
    test_sents=[]
    test_lens=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f.read().split('\n'):
            test_sents.append(line)
            test_lens.append(len(line))
    return test_sents,test_lens
test_sents,test_lens=read_testdata('./data/sample_per_line_preliminary_A.txt')
with open("./data/label_dict_new", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {_id: _label for _label, _id in list(label2id.items())}
test_dataset = NERDataset(test_sents, config,label2id)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                          shuffle=False, collate_fn=test_dataset.collate_fn)

class BertNER(nn.Module):
    def __init__(self,config,label2id):
        super(BertNER, self).__init__()
        self.bert=BertModel.from_pretrained('bert-base-chinese')
        self.num_labels=len(label2id)
        self.classifier=nn.Linear(768,self.num_labels)
        self.crf=CRF(self.num_labels,batch_first=True)

    def forward(self,input_data,label_mask,input_label_starts):
        input_ids,label_mask,input_label_starts = input_data,label_mask,input_label_starts
        attention_mask=input_ids.gt(0)
        token_type_ids = attention_mask.eq(0).int().long()
        outputs=self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_label_starts)]
        padded_sequence_output=pad_sequence(origin_sequence_output,batch_first=True)
        logits=self.classifier(padded_sequence_output)
        output=self.crf.decode(logits,mask=label_mask)
        return output

def predict():
    if os.path.exists(config.model_dir) and config.load_before:
        state = torch.load(config.model_dir)
        model = BertNER(config,label2id)
        model.to(config.device)
        model.load_state_dict(state['model_state'])
        print('predict阶段加载模型完成')
    model.eval()
    with torch.no_grad():
        all_predict_label =[]
        for step,batch in enumerate(tqdm(test_loader)):
            one_predict_label = []
            batch_data,label_mask,batch_label_starts=batch
            batch_predict=model(batch_data,label_mask,batch_label_starts)
            for i in range(len(batch_predict)):
                for j in batch_predict[i]:
                    one_predict_label.append(id2label[j])
                all_predict_label.append(one_predict_label)
                one_predict_label=[]
    assert sum([len(x) for x in all_predict_label])==sum(test_lens)
    return all_predict_label
predict_label=predict()
line_results=[]
for sent,labels in zip(test_sents,predict_label):
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