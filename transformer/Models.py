import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols

# 입력한 각 요소들에 대해서 위치 정보를 인식할 수 있도록 돕는 함수
# 위치 정보가 왜 필요한가?
# 트랜스포머의 경우 시퀀스가 한 번에 병렬로 입력되기 때문에 단어 순서에 대한 정보가 사라짐
# 따라서 단어 위치 정보를 별도로 계산해줘야 함 (position encoding)
# 이때 의미 정보가 변질되지 않도록 위치 벡터값이 너무 크면 안되기 때문에, [-1, 1]을 반복하는 주기 함수인 sin, cos 사용
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    # position과 hid_idx에 대한 angle 계산 -> position encoding의 주기성 결정
    # position: 시퀀스에서의 위치
    # hid_idx: hidden 차원의 인덱스
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    # 특정 위치(position)에 대해 전체 차원에 걸쳐 angle을 계산하고, 벡터 형태로 반환
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    # 각 위치에 대해서 get_posi_angle_vec 함수를 사용해서 계산된 각도 저장하는 배열
    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    # 짝수 인덱스 차원에 sin 함수를, 홀수 인덱스 차원에 cos 함수를 적용
    # -> 각 위치의 encoding 구성
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # 패딩 인덱스가 주어졌다면 해당 인덱스의 position encoding값을 0으로 설정해서 그 위치 무시하도록 함
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

# 트랜스포머의 인코더
class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1                      # 모델이 처리할 수 있는 최대 시퀀스 길이
        n_src_vocab = len(symbols) + 1                              # 입력 데이터의 단어 개수
        d_word_vec = config["transformer"]["encoder_hidden"]        # 각 단어의 hidden 차원. 단어 임베딩에 사용
        n_layers = config["transformer"]["encoder_layer"]           # 인코더 내의 FFTBlock 레이어 수
        # 아래는 모두 FFT block 매개변수로 들어감
        n_head = config["transformer"]["encoder_head"]              # multi head attention에서 head 수
        d_k = d_v = (                                               # d_k, d_v: 어텐션 head에서 key, value 차원
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]           # 모델의 피처 벡터 차원 (=입력 임베딩 차원)
        d_inner = config["transformer"]["conv_filter_size"]         # convolution layer 내부 차원 (FFT 네트워크 내부 차원)
        kernel_size = config["transformer"]["conv_kernel_size"]     # convolution layer 커널 크기
        dropout = config["transformer"]["encoder_dropout"]          # dropout 비율

        # encoder가 처리할 수 있는 최대 시퀀스 길이
        self.max_seq_len = config["max_seq_len"]
        # encoder가 처리할 모델 차원
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )

        # (처리할 수 있는 최대 시퀀스 길이 +1) 크기의 position encoding table 생성
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        # n_layers 만큼 FFTBLock을 생성
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    # 입력 시퀀스를 인코딩한 결과 리턴하는 함수
    # src_seq: 입력 시퀀스
    def forward(self, src_seq, mask, return_attns=False):

        # self attention 가중치를 저장할 리스트 생성
        enc_slf_attn_list = []
        # 입력 시퀀스의 배치 크기와 입력 시퀀스의 최대 길이
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        # self attention 할 때, 현재 단어나 과거 단어 등 어떤 부분을 봐야 하는지, 그리고 미래 단어와 패딩된 부분처럼 무시해야 할 부분을 마스킹
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # 모델이 training 상태가 아니고, 입력 시퀀스의 길이가 처리할 수 있는 최대 길이보다 길 때
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            # 입력 시퀀스 임베딩이랑 sinusoid encoding table 생성한 걸 더함
            # -> 모델이단어와 위치 정보까지 고려할 수 있게 됨
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        # training 상태고, 입력 시퀀스의 길이가 처리할 수 있는 최대 길이보다 짧을 때
        else:
            # 기존의 position encoding를 이용
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        # layer_stack에 저장한 n_layers개의 FFTBLock layer에 대해 forward 수행 (각 레이어가 입력을 처리하고, 다음 레이어로 출력 전달)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            # return_attns가 True일 경우
            # 각 레이어의 self-attention 가중치를 enc_slf_attn_list에 추가
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        # 모든 encoder layer를 통과한 결과
        # decoder로 전달됨
        return enc_output

# 트랜스포머의 디코더
class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    # enc_seq: 인코더에서 준 시퀀스
    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        # 인코더 시퀀스의 배치 크기와 최대 길이
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        # 모델이 training 상태가 아니고, 인코더 시퀀스의 길이가 처리 가능한 최대 길이보다 길 때
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            # 현재 이후의 모든 토큰을 마스킹 해서 디코더가 이전 결과만 참고할 수 있도록 함
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            # 인코더 시퀀스랑 sinusoid encoding table 생성한 걸 더함
            # -> 모델이 단어와 위치 정보까지 고려할 수 있게 됨함
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        # training 상태고, 인코더 시퀀스의 길이가 처리 가능한 최대 길이보다 짧을 때
        else:
            # 처리 가능한 최대 길이와, 실제 최대 길이 중 min
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

            # 인코더 시퀀스에 position encoding을 더함 -> 위치정보까지 고려할 수 있게 됨
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            # mask, slf_attn_mask를 max_len으로 슬라이싱 -> 실제 데이터 길이에 맞추는 작업
            # mask: 시퀀스 내에 패딩된 부분은 실제 데이터가 아니기 때문에 어텐션 계산 할 때 마스킹
            # self attention mask는 디코더가 현재 위치까지의 정보만 사용해야 해서 미래 정보 보지못하게 마스킹
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        # layer_stack에 있는 각 디코더 FFTBlock layer를 forward (각 레이어의 output을 다음 레이어의 input으로 사용)
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            # return_attns가 True일 경우, 각 레이어의 self-attention 가중치를 dec_slf_attn_list에 추가
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        # 모든 디코더 레이어를 통과한 결과
        return dec_output, mask
