from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from utils import preprocess

dataset = preprocess(mode='t')

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
trainer = trainers.WordPieceTrainer(
    vocab_size=53000,  # 얼마나 많은 단어를 기억할거냐
    min_frequency=1,  # 얼마나 자주 등장해야 vocab에 추가할건지
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[START]", "[EOS]"]
)

tokenizer.train_from_iterator(dataset, trainer)

encoded = tokenizer.encode("[START]my name is sieon?[EOS]")
print(encoded.ids)
decoded_string = tokenizer.decode(encoded.ids, skip_special_tokens=False)
print(f'Decoded string -> {decoded_string}')

tokenizer.save("korean_tokenizer.json")

print('Vocab_size : ',tokenizer.get_vocab_size()) # 53000
