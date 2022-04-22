import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

class NERDataset(Dataset):
    def __init__(self,words,labels,config,label2id,word_pad_idx=0,label_pad_idx=-1):
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
        self.label2id=label2id
        self.id2label={_id: _label for _label, _id in list(label2id.items())}
        self.dataset=self.preprocess(words,labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device
    def preprocess(self,origin_sentences,origin_labels):
        data=[]
        sentences=[]
        labels=[]
        for line in origin_sentences:
            n=len(line)
            token_start_idxs=[]
            for i in range(n):
                token_start_idxs.extend([i+1])
            token_start_idxs=np.array(token_start_idxs)
            words=['[CLS]']+line
            sentences.append((self.tokenizer.convert_tokens_to_ids(words),token_start_idxs))
        for tag in origin_labels:
            label_id=[self.label2id.get(t) for t in tag]
            labels.append(label_id)
        for sentence,label in zip(sentences,labels):
            data.append((sentence,label))
        return data
    def __getitem__(self, idx):
        word=self.dataset[idx][0]
        label=self.dataset[idx][1]
        return [word,label]
    def __len__(self):
        return len(self.dataset)

    def collate_fn(self,batch):
        sentences=[x[0] for x in batch]
        labels = [x[1] for x in batch]
        batch_len = len(sentences)
        max_len=max([len(s[0]) for s in sentences])
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts=[]
        for j in range(batch_len):
            cur_len=len(sentences[j][0])
            batch_data[j][:cur_len]=sentences[j][0]
        #去除cls和pad对模型的影响
        for j in range(batch_len):
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)

        batch_labels = self.label_pad_idx * np.ones((batch_len, max_len-1))
        for j in range(batch_len):
            cur_tags_len=len(labels[j])
            batch_labels[j][:cur_tags_len]=labels[j]

        batch_data=torch.tensor(batch_data,dtype=torch.long).to(self.device)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long).to(self.device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

        return [batch_data, batch_label_starts, batch_labels]