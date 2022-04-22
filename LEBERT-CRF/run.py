import torch
from transformers import BertTokenizer,BertConfig
from torch.utils.data import DataLoader
from transformers.optimization import AdamW,get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm
import logging
import time

from utils import set_train_args,seed_everything,build_lexicon_tree_from_vocabs,val_split,set_logger
from data_utils import get_corpus_matched_word_from_lexicon_tree,ItemVocabLabel,ItemVocabArray,load_word_embedding,pretrained_embedding_for_corpus
from model import LEBertCRF
from processor import TaskDataset,NERDataset
from metric import f1_score

args = set_train_args()
seed_everything(args.seed)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.train_file = os.path.join(args.data_path, 'train.npz')
args.label_path = os.path.join(args.data_path, 'labels.txt')

writer = SummaryWriter(os.path.join(args.tensorboard_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
set_logger(args.log_dir)

lexicon_tree = build_lexicon_tree_from_vocabs([args.pretrain_embed_path], scan_nums=[args.max_scan_num])
embed_lexicon_tree = lexicon_tree

matched_words = get_corpus_matched_word_from_lexicon_tree(args.train_file, embed_lexicon_tree)
word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False)#初始化匹配词词表
label_vocab = ItemVocabLabel(files=[args.label_file], is_word=False)
word_embedding, embed_dim=load_word_embedding(args.pretrain_embed_path,args.max_scan_num)
match_word_embedding=pretrained_embedding_for_corpus(word_vocab,word_embedding,embed_dim)#加载匹配词词向量
print('词汇预先匹配完成')

tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)
config = BertConfig.from_pretrained(args.config_name)
config.num_labels = label_vocab.item_size
config.add_layer = 0
config.word_vocab_size = match_word_embedding.shape[0]
config.word_embed_dim = match_word_embedding.shape[1]
# 初始化模型
model = LEBertCRF.from_pretrained(args.pretrain_model_path, config=config).to(args.device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

no_decay = ["bias", "LayerNorm.weight"]
model_param = list(model.named_parameters())
bert_param_optimizer = []
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

#数据集
word_train, word_dev, label_train, label_dev= val_split(args.train_file)
train_features=TaskDataset(word_train,label_train,tokenizer,label_vocab,word_vocab,args,embed_lexicon_tree).data
train_dataset = NERDataset(train_features,args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=train_dataset.collate_fn)
dev_features=TaskDataset(word_dev,label_dev,tokenizer,label_vocab,word_vocab,args,embed_lexicon_tree).data
dev_dataset = NERDataset(dev_features,args)
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dev_dataset.collate_fn)

optimizer = AdamW(optimizer_grouped_parameters, lr=args.classifier_learning_rate,eps=args.eps)
train_size = len(train_dataloader)
total_size=args.epochs * train_size
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0.1 * total_size,
                                            num_training_steps=total_size)


id2label = {_id: _label for _label, _id in list(label_vocab.item2idx.items())}
best_val_f1=0.0

logging.info("--------Start Training!--------")
for epoch in range(1,args.epochs+1):
    train_losses=0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_dataloader)):
        input_ids = data['input_ids'].to(args.device)
        token_type_ids = data['token_type_ids'].to(args.device)
        attention_mask = data['attention_mask'].to(args.device)
        word_ids = data['word_ids'].to(args.device)
        word_mask = data['word_mask'].to(args.device)
        label_ids = data['label_ids'].to(args.device)
        loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask,label_ids)
        train_losses += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_dataloader)
    logging.info('Epoch: {},train loss: {}'.format(epoch, train_loss))
    # tensorboard --logdir "./runs"启动
    writer.add_scalar('Training/training loss', train_loss, epoch)

    model.eval()
    with torch.no_grad():
        true_tags,pred_tags=[],[]
        val_losses,val_f1 = 0.0,0.0
        for batch_idx, data in enumerate(tqdm(dev_dataloader)):
            input_ids = data['input_ids'].to(args.device)
            token_type_ids = data['token_type_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            word_ids = data['word_ids'].to(args.device)
            word_mask = data['word_mask'].to(args.device)
            label_ids = data['label_ids'].to(args.device)
            loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
            val_losses+=loss.item()
            label_mask = label_ids.gt(-1)
            batch_output=model.crf.decode(logits,mask=label_mask)
            batch_tags = label_ids.to('cpu').numpy()
            pred_tags.extend([id2label.get(idx) for idx in indices] for indices in batch_output)
            true_tags.extend([id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags)
        val_loss=float(val_losses) / len(dev_dataloader)
        metrics = {}
        recall, precision, f1= f1_score(true_tags, pred_tags)
        metrics["recall"] = recall
        metrics["precision"] = precision
        metrics["f1"] = f1
        logging.info("Epoch: {}, dev loss: {},dev recall: {},dev precision: {},dev f1score: {}".format(epoch, val_loss,metrics["recall"],metrics["precision"],metrics["f1"]))
        writer.add_scalar('Validation/loss', val_loss, epoch)
        writer.add_scalar('Validation/recall', metrics["recall"], epoch)
        writer.add_scalar('Validation/precision', metrics["precision"], epoch)
        writer.add_scalar('Validation/f1',metrics["f1"], epoch)

        improve_f1 = metrics["f1"] - best_val_f1
        state = {}
        if improve_f1 > 1e-5:
            best_val_f1 = metrics["f1"]
            state["model_state"] = model.state_dict()
            torch.save(state, args.output_path)
            logging.info("--------Save best model!--------")


