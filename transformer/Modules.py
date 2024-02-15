import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        # 내적 결과를 스케일링 하는 데 사용함. 값이 클 수록 softmax 결과가 평탄함. 
        # 각 key의 영향력을 조절해서 q와 관련성이 높은 key를 강조함
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    # 순전파
    # q: Query, k: Key, v: Value, mask: 어텐션 계산할 때 선택적으로 하기 위함
    def forward(self, q, k, v, mask=None):
        # 어텐션 스코어 계산
        # torch.bmm: 배치의 행렬곱 (batch matrix multiplication)
        # Query와 Key의 dot product로 어텐션 스코어 계산
        # temperature로 나눠서 스케일링
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        # mask가 주어졌다면 mask의 true 위치에 -np.inf로 설정해서 softmax 적용하면 어텐션 스코어를 0으로 만듦
        # -> 특정 토큰 무시
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # softmax 함수로 어텐션 가중치를 확률 분포로 변환
        # 각 Key가 Query에 대해 상대적으로 얼마나 중요한지 나타냄
        attn = self.softmax(attn)
        # torch.bmm으로 value 행렬 v와 어텐션 스코어 행렬곱
        output = torch.bmm(attn, v)

        return output, attn
