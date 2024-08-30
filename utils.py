import pandas as pd
import os
from tqdm import tqdm


spetial_tokens = ['[START]','[UNK]','[EOS]','[PAD]']

def preprocess(dir:str = '/media/sien/DATA/DATA/dataset/ko_en_translator_data',mode = 'p'):
    try:
        data_dic = {'input_idx' : [], 'target' : []}
        full_list = []
        for high, _, name in os.walk(dir):
            for spesific in name:
                path = os.path.join(high, spesific)
                full_list.append(path)
        for i in tqdm(full_list, desc= 'execl to dict converting ...'):
            df = pd.read_excel(i)
            origin = df['원문'].values.tolist()
            trans = df['번역문'].values.tolist()
            data_dic['input_idx'].extend(origin)
            data_dic['target'].extend(trans)
        flatten_input_idx = ['[START]' + i + '[EOS]' for i in data_dic['input_idx']]
        flatten_target = ['[START]' + i + '[EOS]' for i in data_dic['target']]
        print(f'total data len is {len(flatten_input_idx)}\n')

        if mode == 't':
            tmp = flatten_input_idx + flatten_target
            return tmp

        return dict(input_idx = flatten_input_idx[:20000], target = flatten_target[:20000]), dict(input_idx = flatten_input_idx[20000:20200], target = flatten_target[20000:20200])
    except Exception as e:
        print(f'error occur : {e.with_traceback()}')

if __name__ == "__main__":
    preprocess()
