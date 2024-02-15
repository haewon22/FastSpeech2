import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    # Scaled Dot-Product Attention을 n_head번 병렬 수행 한 후 나오는 행렬을 모두 concatenate

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
 
        self.n_head = n_head    # 어텐션 헤드 수
        self.d_k = d_k          # key 차원
        self.d_v = d_v          # value 차원

        # 각 쿼리(Q), 키(K), 값(V)에 대해 선형변환
        # nn.Linear: 선형변환(linear transformation)하는 모듈. 입력 벡터에 가중치 행렬 곱하고 편향 더함 (편향: 디폴트 True)
        # 여기서는 d_model 차원 입력을 선형변환해서 n_head * (d_k 또는 d_v) 차원으로 변환하고 각 어텐션 헤드가 처리할 Q, K, V 벡터 생성
        # 이렇게 하면 입력 데이터를 각각 다른 subspace로 매핑하고 multi-head-attention에서 다양한 관점에서 정보 추출 가능하게 됨
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # ScaledDotProductAttention 모듈로 attention 생성
        # temperature: 어텐션 스코어 스케일링을 위함. (루트 d_k)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(d_model)

        # 모든 어텐션 헤드의 출력을 다시 d_model 차원으로 선형 변환
        self.fc = nn.Linear(n_head * d_v, d_model)

        # outfitting 방지를 위한 dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # .size(): 원소 개수
        # sz_b: 배치 사이즈
        sz_b, len_q, _ = q.size()  
        sz_b, len_k, _ = k.size()  
        sz_b, len_v, _ = v.size()  

        # MHA에서 residual connection에 사용하기 위해 저장
        residual = q

        # 선형변환된 q, k, v를 sz_b, len_, n_head, d_에 맞게 텐서 차원을 재구성
        # .view(): 원소 수를 유지하면서 텐서의 shape 변경
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # permute: 모든 차원 맞교환해서 재배열
        # contiguous: 메모리 상에서 텐서의 저장 방식을 연속으로 만듦
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # 마스크를 어텐션 헤드(n_head)만큼 반복해서 복제, 모든 헤드에 같은 마스크를 씌우기 위함
        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        # 어텐션 연산
        # output: 어텐션 적용 결과, attn: 어텐션 가중치
        output, attn = self.attention(q, k, v, mask=mask)

        # 멀티 헤드 어텐션의 각 헤드로부터 얻은 출력을 원래의 텐서 구조로 재구성
        # 멀티 헤드 어텐션에서 각 헤드가 독립적으로 생성한 출력을 적절하게 분리하고, 하나의 텐서로 합침
        # output은 (n_head * sz_b) x len_q x d_v 형태로 모든 어텐션 헤드의 결과가 일렬로 나열 돼있음
        # .view로 output을 (n_head, sz_b, len_q, d_v)로 변경
        # (정확한지 모르겠음)
        output = output.view(n_head, sz_b, len_q, d_v)

        output = (
            # 결과: (sz_b, len_q, n_head * d_v)
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        # output에 선형 변환을 하고, dropout을 적용
        output = self.dropout(self.fc(output))
        # 현재 입력 residual(q)을 현재 출력(output)에 더하고(residual connection 부분), 레이어 정규화
        output = self.layer_norm(output + residual)

        return output, attn

# 각각 적용되는 두 개의 feed forward layer로 구성됨
# 멀티 헤드 어텐션 레이어의 출력에 적용되고, 시퀀스의 각 위치에 있는 피처를 개별적으로 처리
class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()
        # d_in: 입력 차원
        # d_hid: hidden 차원
        # kernel_size: 커널 사이즈
        # dropout: dropout 비율

        # Use Conv1D (w_1, w_2)
        # position-wise
        # PositionsizeFeedForward 네트워크의 두 번째 conv layer
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            # 입력과 출력의 길이가 동일하도록(연산 후에도 길이가 변하지 않도록) 커널 양쪽에 패딩
            padding=(kernel_size[0] - 1) // 2,   
        )
        # position-wise
        # PositionsizeFeedForward 네트워크의 두 번째 conv layer
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            # 입력과 출력의 길이가 동일하도록(연산 후에도 길이가 변하지 않도록) 커널 양쪽에 패딩
            padding=(kernel_size[1] - 1) // 2,
        )

        # 레이어 정규화 layer. 각 feedforward layer의 출력을 정규화
        self.layer_norm = nn.LayerNorm(d_in)
        # dropout layer. 무작위로 선택된 뉴련을 0으로 설정 (제거)함으로써 overfitting 방지
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # residual connection을 위함
        residual = x
        # x의 차원을 transpose 해서 Conv1d에 맞게 함
        output = x.transpose(1, 2)
        # 첫 번째 Conv1d w_1을 적용 -> ReLU 활성화 함수 통과 -> w_2 적용
        output = self.w_2(F.relu(self.w_1(output)))
        # 다시 transpose 해서 원래 차원으로 되돌림
        output = output.transpose(1, 2)
        # dropout 적용
        output = self.dropout(output)
        # 현재 입력 residual(x)을 현재 출력(output)에 더하고(residual connection 부분), 레이어 정규화
        output = self.layer_norm(output + residual)

        return output
