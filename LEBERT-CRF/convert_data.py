
from tqdm import tqdm
import codecs
import numpy as np
import logging
import json

def load_sentences(path):
    sentences, labels = [], []
    sentence,label=[],[]
    dict={}
    item = 0
    for line in tqdm(codecs.open(path,'r',encoding='utf-8')):
        line=line.strip()
        if not line:
            if len(sentence)>0:
                assert len(sentence)==len(label)
                sentences.append(sentence)
                labels.append(label)
                sentence=[]
                label=[]
        else:
            word=line.split(' ')
            if len(word)==2:
                sentence.append(word[0])
                label.append(word[-1])
                if word[-1] not in dict:
                    dict[word[-1]]=item
                    item+=1
            else:
                sentence.append(' ')
                label.extend(word)
                if word[0] not in dict:
                    dict[word[0]]=item
                    item += 1
        #循环走完，要判断一下，防止最后一个句子没有进入到句子集合中
    if len(sentence)>0 and len(label)>0:
        sentences.append(sentence)
        labels.append(label)
    # with open("./dataset/label_dict", "w", encoding='utf-8') as f:
    #     json.dump(dict, f,sort_keys=True, ensure_ascii=False)
    with open("./dataset/labels.txt", "w", encoding='utf-8') as f:
        for key,value in dict.items():
            f.write(key+'\n')
    np.savez_compressed('./dataset/train'+'.npz', words=sentences, labels=labels)
    logging.info("--------ata process DONE!--------")
load_sentences('./dataset/train.txt')
print('finish')