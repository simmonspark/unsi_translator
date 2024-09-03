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
batch_size = 4
compile = False

criterion = nn.CrossEntropyLoss()

torch.manual_seed(1234)
cfg = TSConfig()
model = TS(cfg)

print('===================================================')
print('   시언이의 번역기 autoregressive fine tune.  process  ')
print('===================================================\n')
print('---- Config as follow ----\n')
print(cfg)
scaler = torch.cuda.amp.GradScaler(enabled=True)
print('cuda amp GradScaler at 16bit cal [ON]')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(f'TOKENIZERS_PARALLELISM [OFF]')

t_data, v_data = preprocess(mode='a')
t_dataset = TSDataset(t_data)
v_dataset = TSDataset(t_data)
train_loader = DataLoader(t_dataset, batch_size=batch_size, pin_memory=True, pin_memory_device='cuda')
val_loader = DataLoader(v_dataset, batch_size=batch_size, pin_memory=True, pin_memory_device='cuda')

print('load checkpoint...')
checkpoint = torch.load('out/checkpoint.pt')
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model = model.to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)


@torch.no_grad()
def cal_loss():
    print('\nvalidation start\n')
    model.eval()
    losses = []
    loop = tqdm(val_loader, leave=True)
    for data in loop:
        hiddin_sequence = torch.tensor([4], dtype=torch.long).unsqueeze(0).to('cuda')
        hiddin_sequence = hiddin_sequence.repeat(batch_size, 1)
        x, y, att_mask = data
        x = x.to('cuda')
        y = y.to(torch.long).to('cuda')
        key_pad_mask = att_mask.to('cuda').unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            all_logits = []
            for i in range(max_token):
                encoder_out = model.encoder(x, attention_mask=key_pad_mask)
                attn_mask = model.make_trg_mask(hiddin_sequence)
                decoder_out = model.decoder(hiddin_sequence, encoder_out, key_pad_mask, attn_mask)
                logit = model.fc_out(decoder_out)
                all_logits.append(logit[:, -1:, :])
                next_token = torch.argmax(logit[:, -1, :], dim=-1)
                next_token = next_token.unsqueeze(1)
                hiddin_sequence = torch.cat((hiddin_sequence, next_token), dim=-1)
            all_logits = torch.cat(all_logits, dim=1)
            loss = criterion(all_logits.view(-1, cfg.vocab_size), y.view(-1))
        losses.append(loss)
        loop.set_postfix(loss=loss.item())
    model.train()
    print(f'val loss : {sum(losses) / len(losses)}')
    return sum(losses) / len(losses)


max_token = 128
best = 1e9

for iter in range(epoch):
    g_loss = []
    loop = tqdm(train_loader, leave=True)
    for data in loop:
        x, y, att_mask = data
        x = x.to('cuda')
        y = y.to(torch.long).to('cuda')
        key_pad_mask = att_mask.to('cuda').unsqueeze(1)
        hiddin_sequence = torch.tensor([4], dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to('cuda')
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            encoder_out = model.encoder(x, attention_mask=key_pad_mask)
            all_logits = []
            for i in range(max_token):
                attn_mask = model.make_trg_mask(hiddin_sequence)
                decoder_out = model.decoder(hiddin_sequence, encoder_out, key_pad_mask, attn_mask)
                logit = model.fc_out(decoder_out)
                all_logits.append(logit[:, -1:, :])
                next_token = torch.argmax(logit[:, -1, :], dim=-1).unsqueeze(1)
                hiddin_sequence = torch.cat((hiddin_sequence, next_token), dim=-1)
            all_logits = torch.cat(all_logits, dim=1)
            loss = criterion(all_logits.view(-1, cfg.vocab_size), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        g_loss.append(loss.item())
        loop.set_postfix(loss=loss.item())
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
