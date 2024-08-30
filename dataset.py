from torch.utils.data import Dataset,DataLoader
import torch
from tokenizers import Tokenizer

tokenizer_path = "korean_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

max_length = 512

'''
example as below

attention mask : trg mask
key pad mask : 0000...1111...1
sequence pad : [START] 123 345 456 3 3 3 .... [EOS]

'''


def add_padding(ids, max_length=max_length, pad_id=3):
    if len(ids) < max_length:
        return ids + [pad_id] * (max_length - len(ids))
    return ids[:max_length]


class TSDataset(Dataset):
    def __init__(self, data_dic):
        super().__init__()
        self.data_dic = data_dic

    def __getitem__(self, item):
        input_data, label = self.data_dic['input_idx'][item], self.data_dic['target'][item]

        input_data = tokenizer.encode(input_data)

        key_pad_mask = add_padding([1] * len(input_data.ids), pad_id=0)

        input_data = add_padding(input_data.ids)

        label = tokenizer.encode(label)

        label = add_padding(label.ids)

        return torch.Tensor(input_data).type(torch.int), torch.Tensor(label).type(torch.int), torch.Tensor(
            key_pad_mask).type(torch.bool)

    def __len__(self):
        return len(self.data_dic['input_idx'])



if __name__ == "__main__":
    from utils import preprocess

    t_data, v_data = preprocess()
    dataset = TSDataset(t_data)
    x, y, a = next(iter(dataset))
    print()
    # flatten_input = [item for sublist in data['input_data'] for item in sublist]
    # flatten_label = [item for sublist in data['label'] for item in sublist]
    test_loader = DataLoader(dataset, batch_size=8, pin_memory=True, pin_memory_device='cuda')
    while True:
        print(next(iter(test_loader)))

