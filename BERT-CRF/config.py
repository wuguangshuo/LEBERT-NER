import torch

device='cuda' if torch.cuda.is_available() else 'cpu'
log_dir='./log/train.log'
val_split_size=0.2

train_dir='./data/train.npz'
train_dir='./data/train_new.npz'
batch_size=2
epoch_num=11
model_dir = './save/'+ 'model' + '.pkl'
load_before=True
patience_num=3
min_epoch_num=5
patience=1e-3
dropout=0.2
clip_grad=5