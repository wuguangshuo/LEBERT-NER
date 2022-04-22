import logging
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import torch
import torch.nn as nn

import config
from model import BertNER
from metric import f1_score
from attack import PGD, FGM



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


def val_split(dataset_dir):
    """split one dev set without k-fold"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"].tolist()[:50]
    labels = data["labels"].tolist()[:50]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.val_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev

def train(train_loader,val_loader,model,optimizer, scheduler,label2id,open_ad='pgd'):
    if os.path.exists(config.model_dir) and config.load_before:
        state = torch.load(config.model_dir)
        model = BertNER(config,label2id)
        model.to(config.device)
        model.load_state_dict(state['model_state'])
        print('train阶段加载模型完成')
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(1,config.epoch_num+1):
        model.train()
        train_losses=0
        for idx,batch_samples in enumerate(tqdm(train_loader)):
            batch_data,batch_token_starts,batch_labels=batch_samples
            batch_masks=batch_data.gt(0)
            loss=model([batch_data,batch_token_starts],attention_mask=batch_masks, labels=batch_labels)[0]
            train_losses += loss.item()
            # clear previous gradients, compute gradients of all variables wrt loss
            model.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)


            optimizer.step()
            scheduler.step()
        train_loss=float(train_losses)/len(train_loader)
        logging.info('Epoch: {},train loss: {}'.format(epoch,train_loss))
        val_metrics = evaluate(val_loader, model,label2id)
        val_f1 = val_metrics['f1']
        print("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
        improve_f1 = val_f1 - best_val_f1
        state = {}
        if improve_f1 >1e-5:
            best_val_f1 = val_f1
            state['model_state'] = model.state_dict()
            torch.save(state,config.model_dir)
            logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
    logging.info("Training Finished!")

def evaluate(val_loader,model,label2id):
    model.eval()
    true_tags=[]
    pred_tags = []
    val_losses = 0

    with torch.no_grad():
        id2label = {_id: _label for _label, _id in list(label2id.items())}
        for idx,batch_samples in enumerate(val_loader):
            batch_data,batch_token_starts,batch_labels=batch_samples
            label_masks = batch_labels.gt(-1)
            batch_masks=batch_data.gt(0)
            loss,logits=model([batch_data,batch_token_starts],attention_mask=batch_masks, labels=batch_labels)
            val_losses+=loss.item()
            batch_output=model.crf.decode(logits,mask=label_masks)
            batch_tags = batch_labels.to('cpu').numpy()
            pred_tags.extend([id2label.get(idx) for idx in indices] for indices in batch_output)
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags)
        metrics = {}
        recall, precision, f1= f1_score(true_tags, pred_tags)
        metrics['recall'] = recall
        metrics['precision'] = precision
        metrics['f1'] = f1
        metrics['loss'] = float(val_losses) / len(val_loader)
        return metrics