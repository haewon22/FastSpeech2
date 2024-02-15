import os
import json 
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""
    """
    aims to add varianve information (duration, pitch, energy) to the phoneme hidden sequence
    which can provide enough information to predict variant speech for the one-to-many problems in TTS
    3-variance information: 1) phoneme duration, 2) pitch, 3) energy
    """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        # 지속시간 예측을 위한 인스턴스 VariancePredictor 생성
        self.duration_predictor = VariancePredictor(model_config)
        # 길이 조절을 위한 인스턴스 LengthRegulator 생성
        self.length_regulator = LengthRegulator()
        # 피치 예측을 위한 인스턴스 VariancePredictor 생성
        self.pitch_predictor = VariancePredictor(model_config)
        # 에너지 예측을 위한 인스턴스 VariancePredictor 생성
        self.energy_predictor = VariancePredictor(model_config)

        # 피치가 처리되는 수준 ("phoneme_level", "frame_level")
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        # 에너지가 처리되는 수준 ("phoneme_level", "frame_level")
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        # 양자화: 연속 -> 이산. 모델에서 처리하기 쉬운 상태로 가공하는 거임
        # 피치 양자화 종류 ("linear", "log")
        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        # 에너지 양자화 종류 ("linear", "log")
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        # 양자화 될 때 몇 개의 구간으로 나눌건지
        n_bins = model_config["variance_embedding"]["n_bins"]
        # log 양자화: log 씌우고 나누고 다시 exp로 되돌리기 때문에 작은 값들 사이 구간은 더 작고, 큰 값들 사이 구간은 더 넒어짐. 
        #            -> 상대적인 비율을 유지하면서 구간 나눌 수 있음
        # linear 양자화: 동일한 간격으로 나눔 (0~100을 10씩 10개로)
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        # 전처리된 파일이 있는 경로 열기
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            # 피치 최소, 최댓값 저장
            pitch_min, pitch_max = stats["pitch"][:2]
            # 에너지 최소, 최댓값 저장
            energy_min, energy_max = stats["energy"][:2]

        # 피치와 에너지 값을 양자화 할 bins(구간) 설정
        # 피치 양자화 방식이 log라면
        if pitch_quantization == "log":
            # torch.linspace(start, end, steps): start(pitch_min)와 end(pitch_max) 사이를 step(n_bins - 1)만큼 동일한 간격으로 나눈 숫자 생성
            # log로 스케일링 된 수를 exp로 다시 복원
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        # 피치 양자화 방식이 linear라면
        else:
            # linear하게 pitch_min와 pitch_max 사이를 n_bins - 1 개의 구간으로 나눔
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        # 에너지 양자화 방식이 log일 때도 피치와 동일한 연산
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        # 에너지 양자화 방식이 linear일 때도 피치와 동일한 연산
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        # nn.Embedding: 임베딩을 생성하는 레이어
        # 피치값 임베딩 (n_bins: 임베딩을 위해 분할된 피치 개수, ["transformer"]["encoder_hidden"]: 임베딩 차원)
        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        # 에너지값 임베딩
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    # x: 입력, target: 실제 피치 값(학습 할 때 씀), control: 피치 값 스케일 조정에 사용
    def get_pitch_embedding(self, x, target, mask, control):
        # pitch_predictor로 x의 피치값 예측
        prediction = self.pitch_predictor(x, mask)
        # 실제 피치 값이 주어졌다면 == train이라면
        if target is not None:
            # target값들을 pitch_bins구간으로 변환 -> 각 taget이 각 구간의 인덱스로 매핑됨
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        # 실제 피치 값이 주어지지 않았다면 == inference라면
        else:
            # 예측된 피치 predection값을 control로 스케일링
            prediction = prediction * control
             # 스케일링된 prediction 값들을 pitch_bins구간으로 변환 -> 각 prediction이 각 구간의 인덱스로 매핑됨
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    # x: 입력, target: 실제 에너지 값(학습 할 때 씀), control: 에너지 값 스케일 조정에 사용
    def get_energy_embedding(self, x, target, mask, control):
        # energy_predictor로 x의 에너지값 예측
        prediction = self.energy_predictor(x, mask)
        # train
        if target is not None:
            # 각 target을 energy_bins 구간에 인덱스로 매핑
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        # inference
        else:
            # prediction 스케일링
            prediction = prediction * control
            # 각 target을 energy_bins 구간에 인덱스로 매핑
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        # 입력 feature 벡터
        x,
        # 소스 마스크. x 시퀀스 중 유효한 부분만 처리하기 위함
        src_mask,
        # 멜 스펙트로그램 마스크
        mel_mask=None,
        # 입력 시퀀스의 최대 길이
        max_len=None,
        # 타겟값. 학습 할 때 실제 데이터로 사용
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        # 각 피처(피치, 에너지, 지속시간)를 조정하는 데 사용
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        # duration_predictor로 지속시간 예측
        log_duration_prediction = self.duration_predictor(x, src_mask)
        # 각 음소에 피치 변조를 할 것이라면
        if self.pitch_feature_level == "phoneme_level":
            # 피치 예측값과 임베딩 저장
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
    
            # 입력 피처 벡터x에 피치 정보를 추가
            x = x + pitch_embedding
        # 각 음소에 에너지 변조를 할 것이라면
        if self.energy_feature_level == "phoneme_level":
            # 에너지 예측값과 임베딩 저장
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            # 입력 피처 벡터x에 에너지 정보를 추가
            x = x + energy_embedding

        # 실제 duration 값이 주어졌다면 == train이라면
        if duration_target is not None:
            # length_regulator로 입력 x를 duration 타겟값에 따라 길이 조정
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            # duration_target 그대로 리턴
            duration_rounded = duration_target
        # inference 라면
        else:
            # 로그 스케일된 duration 예측값을 exp로 실제 지속시간으로 복원하고, 지속시간 조정 후 음수가 되지 않도록 조절함
            # torch.clamp(input, min): 최소가 min(0)값이 되도록 그 이하 값(음수)들을 min(0)으로 바꿈
            # log로 예측된 log_duration_prediction를 exp로 다시 복원 후, -1 (로그로 변환할 때 1 더해줬음)이 값을 반올림
            # d_control을 곱해줌으로써 지속시간 조절
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            # 조절된 duration_rounded로 x의 length를 regulation
            # 조절된 mel_len 저장
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            # 조절된 mel_len에 맞게 mel_mask 생성, 유효한 부분만 보기 위함
            mel_mask = get_mask_from_lengths(mel_len)

        # 각 프레임에 피치 변조를 할 것이라면
        if self.pitch_feature_level == "frame_level":
            # 피치 예측값과 임베딩값 저장
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            # 입력 피처 벡터x에 피치 정보를 추가
            x = x + pitch_embedding
        # 각 프레임에 에너지 변조를 할 것이라면
        if self.energy_feature_level == "frame_level":
            # 에너지 예측값과 임베딩 저장
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            # 입력 피처 벡터x에 에너지 정보를 추가
            x = x + energy_embedding

        return (
            x,                        # 변조 후 입력 피처 (피치, 에너지 임베딩 더해진 값)
            pitch_prediction,         # 피치 예측값. 학습 할 때 피치 타겟이랑 비교되거나, 추론 단계에서 음성에 피치 변조할 때 씀 (맞나?)
            energy_prediction,        # 에너지 예측값. 피치 예측값과 설명 동일
            log_duration_prediction,  # 로그 스케일링 된 지속시간 예측값. 
            duration_rounded,         # 반올림된 지속시간 값. train 단계에서는 target값 그대로임
            mel_len,                  # 길이 조절 후 멜 스펙트로그램 길이
            mel_mask,                 # 길이 조절 후 유효한 부분의 멜 스펙트로그램 마스크
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""
    """
    is used to solve the problem of length mismatch 
    between the phoneme and spectrogram squence in the Feed-Forward Transformer
    as well as to control the voice speed and part of prosody
    -> mel-spectrogram sequence 길이와 해당하는 phoneme duration 길이를 맞춤
    """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        # 조절된 입력을 저장할 리스트
        output = list()
        # 조절된 입력의 길이를 저장할 리스트
        mel_len = list()
        
        for batch, expand_target in zip(x, duration):
            # expand()로 현 batch를 expand_target에 맞춰 조절함
            expanded = self.expand(batch, expand_target)
            # 조절한 값을 output 추가
            output.append(expanded)
            # expanded 길이를 mel_len에 추가
            mel_len.append(expanded.shape[0])

        # max_len이 주어졌다면 max_len에 맞춰 패딩
        if max_len is not None:
            output = pad(output, max_len)
        # max_len이 주어지지 않았다면 그냥 패딩
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        # 확장시킨 벡터를 저장할 리스트
        out = list()

        for i, vec in enumerate(batch):
            # 확장된 사이즈 가져옴
            expand_size = predicted[i].item()
            # vec을 expand_size만큼 복제
            out.append(vec.expand(max(int(expand_size), 0), -1))
        # out에 저장된 확장된 벡터들을 
        # torch.cat: concatenate. 주어진 차원에서 텐서 시퀀스를 연결함
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor
    1) Conv1D + ReLU => LN + Dropout => 2) Conv1D + ReLU => LN + Dropout => 3) Linear Layer
    """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        # 입력 피처 벡터 크기 설정
        self.input_size = model_config["transformer"]["encoder_hidden"]
        # convolution 레이어의 필터 크기 설정
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        # convolution 레이어의 커널 크기 설정
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        # convolution 레이어 출력 크기를 필터 크기와 맞춤
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        # 드롭아웃 레이어에서 사용될 드롭아웃 비율
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            # 순서 있는 딕셔너리
            OrderedDict(
                [
                    # 1) Conv1D + ReLU => LN + Dropout
                    (
                        # 첫 번째 1D conv 레이어
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    # 첫 번째 1D conv 레이어 출력에 적용되는 활성화 함수
                    ("relu_1", nn.ReLU()),
                    # 첫 번째 1D conv 레이어 출력 정규화
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    # 첫 번째 드롭아웃 레이어
                    ("dropout_1", nn.Dropout(self.dropout)),

                    # 2) Conv1D + ReLU => LN + Dropout
                    (
                        # 두 번째 1D conv 레이어
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    # 활성화 함수
                    ("relu_2", nn.ReLU()),
                    # 레이어 정규화
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    # 두 번째 드롭아웃 레이어
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        # 3) Linear Layer
        # nn.Linear(self.conv_output_size, 1): conv_output_size를 입력 크기로 받아서 선형변환 후 하나의 출력 생성
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        # 입력 encoder_output을 convolution layer 통과시킴
        out = self.conv_layer(encoder_output)
        # linear layer 통과
        out = self.linear_layer(out)
        # 텐서의 마지막 차원이 1인 차원 제거
        out = out.squeeze(-1)

        # 마스크가 주어졌다면 마스크 처리
        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,      # 입력 채널 차원. conv 연산 입력 벡터 크기
        out_channels,     # 출력 채널 차원. conv 연산 후 벡터 크기
        kernel_size=1,    # 커널 크기. 필터 크기
        stride=1,         # stride 크기
        padding=0,        # 패딩 크기
        dilation=1,       # 커널 사이 요소들 간격
        bias=True,        # 레이어에 편향 추가할 건지
        w_init="linear",  # weight init 종류
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    # 
    def forward(self, x):
        # conv연산에 맞는 형태로 x를 transpose
        x = x.contiguous().transpose(1, 2)
        # convolution 연산
        x = self.conv(x)
        # 다시 transpose 복원
        x = x.contiguous().transpose(1, 2)

        return x
