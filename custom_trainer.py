from typing import Dict, Union, Any
from nets.Model import TS, TSConfig
import torch
from torch import nn
from utils import preprocess
from dataset import TSDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

mode = 'resume'
lr = 5e-6
betas = (0.9, 0.95)
epoch = 100
batch_size = 32
compile = False

criterion = nn.CrossEntropyLoss()

torch.manual_seed(1234)
cfg = TSConfig()
model = TS(cfg)

print('======================================')
print('       시언이의 번역기 train process      ')
print('======================================\n')
print("Initializing a new model from scratch")
print("defaulting to vocab_size 53000 (53004 rounded up for efficiency)\n")
print('---- Config as follow ----\n')
print(cfg)
scaler = torch.cuda.amp.GradScaler(enabled=True)
print('cuda amp GradScaler at 16bit cal [ON]')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(f'TOKENIZERS_PARALLELISM [OFF]')

t_data, v_data = preprocess()
t_dataset = TSDataset(t_data)
v_dataset = TSDataset(t_data)
train_loader = DataLoader(t_dataset, batch_size=batch_size, pin_memory=True, pin_memory_device='cuda')
val_loader = DataLoader(v_dataset, batch_size=batch_size, pin_memory=True, pin_memory_device='cuda')

if mode == 'resume':
    checkpoint = torch.load('out/checkpoint.pt')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to('cuda')
elif mode == 'scratch':
    print('\ntrain from scratch\n')
    model = model.to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)

if compile:
    print("compiling the model... (시간이 좀 걸려요..)")
    unoptimized_model = model
    model = torch.compile(model)
    model = model.to('cuda')


@torch.no_grad()
def cal_loss():
    print('\nvalidation start\n')
    model.eval()
    losses = []
    for D in tqdm(val_loader):
        x, y, att_mask = D
        x = x.to('cuda')
        y = y.to(torch.long).to('cuda')
        att_mask = att_mask.to('cuda')
        with torch.cuda.amp.autocast(enabled=True):
            if mode.startswith('fine'):
                pred = model(input_ids=x, labels=y, attention_mask=att_mask)
                loss = pred.loss
            else:
                pred = model(x, y)
                pred = pred.view(-1, cfg.vocab_size)
                y = y.view(-1)
                loss = criterion(pred, y)
        losses.append(loss)
    model.train()
    print(f'val loss : {sum(losses) / len(losses)}')
    return sum(losses) / len(losses)

best = 1e9
for iter in range(epoch):

    g_loss = []
    for D in tqdm(train_loader):
        x, y, att_mask = D
        x = x.to('cuda')
        y = y.to(torch.long).to('cuda')
        att_mask = att_mask.to('cuda').unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):

            if mode.startswith('fine'):
                pred = model(input_ids=x, labels=y, attention_mask=att_mask)
                loss = pred.loss
            else:
                pred = model(x, y, att_mask)
                pred = pred.view(-1, cfg.vocab_size)
                y = y.view(-1)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        g_loss.append(loss.item())
    print(f"iter {iter}: loss {sum(g_loss) / len(g_loss):.4f}")
    val_loss = cal_loss()
    print(f"Validation loss: {val_loss:.4f}")
    if val_loss < best:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join('out', 'checkpoint.pt'))
        print('saved_checkpoint!\n')
        best = val_loss
