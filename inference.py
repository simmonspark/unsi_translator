import torch
from transformers import T5ForConditionalGeneration
from dataset import add_padding
from utils import *
from nets.Model import TS, TSConfig
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import os
from dataset import TSDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
cfg = TSConfig()
model = TS(cfg)
checkpoint = torch.load('out/checkpoint.pt')
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model = model.to('cuda')

tokenizer_path = "korean_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)


mode = 'reg'

if mode == 'seq':
    t_data, v_data = preprocess(mode = 'a')
    t_dataset = TSDataset(t_data)
    v_dataset = TSDataset(t_data)
    train_loader = DataLoader(t_dataset, batch_size=8, pin_memory=True, pin_memory_device='cuda', shuffle=True)
    val_loader = DataLoader(v_dataset, batch_size=8, pin_memory=True, pin_memory_device='cuda', shuffle=True)

    with torch.no_grad():
        source, target, attention_mask = next(iter(train_loader))

        input_token = source
        output_logits = model(source=source.to('cuda'), target=target.to('cuda'),
                              attention_mask=attention_mask.to('cuda').unsqueeze(1))

        predicted_sequence = torch.argmax(output_logits, dim=-1)

        for _ in range(8):
            decoded_source = tokenizer.decode(input_token[_].tolist(), skip_special_tokens=True)
            decoded_pred = tokenizer.decode(predicted_sequence[_].tolist(), skip_special_tokens=True)
            decoded_string = tokenizer.decode(target[_].tolist(), skip_special_tokens=True)
            print(f'입력은 다음과 같습니다. --> {decoded_source}')
            print(f'대답은 다음과 같습니다. --> {decoded_pred}\n')
            # print(f'정답은 다음과 같습니다. --> {decoded_string}')

if mode == 'reg':
    print('Autoregressive active')
    print('enter any user input ...\n')
    usr_input = input().strip()
    print('Input Sentence : ', usr_input)

    max_token = 50
    hiddin_sequence = torch.tensor([4], dtype=torch.long).unsqueeze(0).to('cuda')
    encoded = torch.tensor(add_padding(tokenizer.encode(usr_input).ids), dtype=torch.long).unsqueeze(0).to('cuda')
    L = len(tokenizer.encode(usr_input).ids)

    with torch.no_grad():
        key_pad_mask = add_padding([1] * L, pad_id=0)
        key_pad_mask = torch.tensor(key_pad_mask, dtype=torch.bool).to('cuda').unsqueeze(0).unsqueeze(0)

        for i in range(max_token):
            encoder_out = model.encoder(encoded, attention_mask=key_pad_mask)
            attn_mask = model.make_trg_mask(hiddin_sequence)

            decoder_out = model.decoder(hiddin_sequence, encoder_out, key_pad_mask, attn_mask)
            logit = model.fc_out(decoder_out)
            next_token = logit[:, -1, :]
            next_token = torch.argmax(next_token, dim=-1)

            next_token = next_token.unsqueeze(0)
            hiddin_sequence = torch.cat((hiddin_sequence, next_token), dim=-1)

            if next_token.item() == 5 or next_token.item() == 3:
                break

        decoded_pred = tokenizer.decode(hiddin_sequence[0].tolist(), skip_special_tokens=True)
        print(decoded_pred)