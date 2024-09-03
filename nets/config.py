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
#사드 ##를 배치 ##하면 그 지역의 땅 ##값이 오르 ##지 않고 , 이런 시설 때문에 그 지역에 피해가 크게 발생하기 때문입니다 .
#$삼성  ##역까지 지하철로 42 ##분 정도 걸려 .