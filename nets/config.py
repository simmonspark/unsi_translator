from dataclasses import dataclass


@dataclass
class TSConfig:
    '''
    : define
     -> 모델에 관련한 파라미터를 정의합니다.
    '''
    block_size: int = 128 # same as max sequence length
    vocab_size: int = 53000
    encoder_layer_n: int = 6
    attention_head_n: int = 8
    decoder_layer_n: int = 6
    embd_dim: int = 768
    bias: bool = False
