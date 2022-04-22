from argparse import ArgumentParser
import logging
from torch.utils.data import DataLoader
from transformers.optimization import AdamW,get_cosine_schedule_with_warmup
import torch
import os
import numpy as np
import random
import json

from utils import set_logger,val_split,train
import config
from data_utils import NERDataset
from model import BertNER

parser = ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--crf_learning_rate', type=float, default=5e-3)
parser.add_argument('--classifier_learning_rate', type=float, default=5e-5)
args = parser.parse_args()

#设置日志
set_logger(config.log_dir)
logging.info('device: {}'.format(config.device))
logging.info('----------process done!-----------')

#设置随机种子
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(2000)

# 将训练集划分训练集和验证集
word_train, word_val, label_train, label_val= val_split(config.train_dir)
with open("./data/label_dict_new", encoding="utf-8") as f:
    label2id = json.load(f)
#构造数据集
train_dataset = NERDataset(word_train, label_train, config,label2id)
val_dataset = NERDataset(word_val, label_val, config,label2id)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                          shuffle=True, collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=val_dataset.collate_fn)

#模型
model = BertNER(config,label2id)
model.to(config.device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

no_decay = ["bias", "LayerNorm.weight"]
model_param = list(model.named_parameters())
bert_param_optimizer = []
bilstm_param_optimizer = []
crf_param_optimizer = []
classifier_param_optimizer = []

for name, param in model_param:
    space = name.split('.')
    if space[0] == 'bert':
        bert_param_optimizer.append((name, param))
    elif space[0] == 'crf':
        crf_param_optimizer.append((name, param))
    elif space[0] == 'classifier':
        classifier_param_optimizer.append((name, param))

optimizer_grouped_parameters = [
    {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01, 'lr': args.learning_rate},
    {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0, 'lr': args.learning_rate},

    {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01, 'lr': args.crf_learning_rate},
    {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0, 'lr': args.crf_learning_rate},

    {"params": [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01, 'lr': args.classifier_learning_rate},
    {"params": [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0, 'lr': args.classifier_learning_rate}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.classifier_learning_rate)
train_size = len(train_dataset)
train_steps_per_epoch = train_size // config.batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0.1 * train_steps_per_epoch,
                                            num_training_steps=config.epoch_num * train_steps_per_epoch)

# Train the model
logging.info("--------Start Training!--------")
train(train_loader, val_loader, model, optimizer, scheduler,label2id)
print('训练完成')
