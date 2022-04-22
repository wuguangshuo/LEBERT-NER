from transformers import BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch

class BertNER(nn.Module):
    def __init__(self,config,label2id):
        super(BertNER, self).__init__()
        self.bert=BertModel.from_pretrained('bert-base-chinese')
        self.num_labels=len(label2id)
        self.classifier=nn.Linear(768,self.num_labels)
        self.crf=CRF(self.num_labels,batch_first=True)
    def forward(self,input_data,attention_mask=None, labels=None):
        input_ids,input_token_starts = input_data
        token_type_ids=attention_mask.eq(0).int().long()
        outputs=self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output=pad_sequence(origin_sequence_output,batch_first=True)
        logits=self.classifier(padded_sequence_output)
        outputs = (logits,)
        loss_mask=labels.gt(-1)
        loss=self.crf(logits,labels,loss_mask)*(-1)
        outputs = (loss,) + outputs
        return outputs
