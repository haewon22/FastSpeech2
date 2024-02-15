from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(torch.nn.Module):
    """FFT Block"""
    """FeedForward Transformer"""

    # d_model: 모델의 차원, n_head: 어텐션 헤드 수, d_k: 키 차원, d_v: value 차원, d_inner: FF 네트워크 내부 차원, kernel_size, dropout: 드롭아웃 비율
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        # 입력 데이터에 대해 multi head attention 수행
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # position-wise한 FeedForward 수행
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    # enc_input: 인코더의 입력
    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        # MultiHeadAttention 결과로 어텐션 결과(enc_output), 가중치(enc_slf_attn) 설정
        # slf_attn_mask: 특정 위치가 어텐션 계산에서 제외되도록 함
        # enc_input이 Query, Key, Value로 모두 사용됨
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        # 마스크가 있다면 마스크된 위치를 0으로 설정 -> 특정 위치 무시
        # mask.unsqueeze(-1): 마스크 차원을 늘려 enc_output에 맞춤
        # masked_fill: 마스크가 True인 위치를 0으로 채워서 enc_output 업데이트
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        # PositionwiseFeedForward 수행 -> 각 위치에서 독립적으로 피드포워드 연산
        enc_output = self.pos_ffn(enc_output)
        # ffn을 적용 하고도 다시 마스크가 있다면 마스크된 위치를 0으로 설정 -> 특정 위치 무시
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        # enc_output: 멀티헤드어텐션 적용 및 position-wise ffn 거친 결과
        # enc_slf_attn: MultiHeadAttention으로 계산된 어텐션 가중치
        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,             # Conv1d의 입력 채널 수
        out_channels,            # Conv1d의 출력 채널 수
        kernel_size=1,           # 커널 크기. Conv 연산에 사용될 윈도우 크기
        stride=1,                # 필터를 적용할 때 입력 데이터를 얼마나 건너뛸 지. =1이므로 하나씩 이동
        padding=None,            # 적용할 패딩 크기 ((kernel_size - 1) / 2)
        dilation=1,              # 커널 사이 요소들 간격. =1은 dilation 없음 의미
        bias=True,               # conv layer에 편향 줄지 여부
        w_init_gain="linear",    # weight init 시에 사용할 gain 종류
    ):
        # ConvNorm: 1d conv 연산 파이토치 모듈
        super(ConvNorm, self).__init__()

        # 패딩을 얼마나 할 지 정해져있지 않을 때 자동으로 계산하기 위함
        if padding is None:
            # 커널 사이즈가 홀수인지 확인
            assert kernel_size % 2 == 1
            # 입력 길이가 conv연산 후에도 변하지 않도록 패딩 값 계산 (출력크기와 입력 크기를 동일하게)
            padding = int(dilation * (kernel_size - 1) / 2)

        # conv1d layer 
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        # 입력된 signal에 1d conv 연산 적용
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    PostNet은 멜 스펙트로그램을 입력으로 받아서 후처리하는 1D Conv layer 집합
    """

    def __init__(
        self,
        n_mel_channels=80,            # 멜 스펙트로그램 채널 수 (입력 데이터 차원)
        postnet_embedding_dim=512,    # conv layer의 임베딩 차원 (각 convolution layer 출력 차원)
        postnet_kernel_size=5,        # conv 커널 크기
        postnet_n_convolutions=5,     # conv layer 수
    ):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        # 1st convolution layer
        # 첫 번째 레이어를 convolutions에 append
        self.convolutions.append(
            # nn.Sequential: 여러 모듈을 하나의 모듈로 묶어 순차적으로 실행할 수 있게 해주는 파이토치 컨테이너
            nn.Sequential(
                # 멜스펙트로그램 채널에서 임베딩 차원으로 매핑하는 convolution 연산 수행
                ConvNorm(
                    n_mel_channels, 
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                # 첫 번째 레이어 출력 정규화
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        # 중간 convolution layer
        # 첫 번째와 마지막 레이어를 제외한 나머지 레이어들을 convolutions에 append
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim, # 입력과 출력 차원이 postnet_embedding_dim로 동일
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2), 
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    # 각 레이어의 출력을 정규화
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        # 마지막 convolution layer
        # 마지막 레이어를 convolutions에 append
        self.convolutions.append(
            nn.Sequential(
                # postnet_embedding_dim에서 원래의 n_mel_channels로 차원 축소
                # -> 최종 결과가 원래의 멜스펙트로그램과 같은 차원을 갖도록 함
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                # 마지막 레이어 출력 정규화
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        # 입력 x를 transpose해서 conv 연산에 맞게 바꿈
        x = x.contiguous().transpose(1, 2)

        # 마지막 레이어를 제외한 레이어에 convolution연산, 활성화 함수(tahn) 적용, dropout 적용
        # tanh: 활성화 함수. 출력을 -1과 1사이로 조정
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        # 마지막 레이어에는 활성화 함수(tanh)를 적용하지 않고 dropout만 적용
        # -> 최종 결과가 원래의 멜 스펙트로그램과 비슷한 형태를 유지하도록 하는 것
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        # x를 다시 transpose해서 차원 복원
        x = x.contiguous().transpose(1, 2)
        return x
