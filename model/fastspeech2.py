import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

class FastSpeech2(nn.Module):
    """ FastSpeech2 """
    """
    phoneme enbedding => positional encoding => encoder => varianve adaptor
    => positional encoding => mel-spectrogram decoder
                           => waveform decoder => fastspeech2s
    """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        # Encoder 인스턴스 생성
        self.encoder = Encoder(model_config)
        # VarianceAdaptor: 피치, 에너지, 지속시간의 variance 모델링 및 조절. 
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        # Decoder 인스턴스 생성
        self.decoder = Decoder(model_config)
        # 디코터의 출력을 멜 스펙트로그램 차원으로 매핑하는 linear layer
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        # PostNet 인스턴스 생성
        self.postnet = PostNet()

        # speaker embedding
        self.speaker_emb = None
        # 여러명의 speaker를 사용한다면 몇 명인지 load해서 n_speaker에 저장
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))

            # n_speaker만큼의 화자에 대해서 임베딩 벡터 생성
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,          # speaker ID
        texts,             # 입력 텍스트 시퀀스
        src_lens,          # 입력 텍스트 시퀀스의 길이
        max_src_len,       # 배치 내에서 가장 긴 입력 길이
        mels=None,         # 학습 시 사용되는 타겟 멜 스펙트로그램
        mel_lens=None,     # 각 타겟 멜 스펙트로그램 길이
        max_mel_len=None,  # 배치 내에서 가장 긴 멜 스펙트로그램 길이
        p_targets=None,    # 피치 타겟값
        e_targets=None,    # 에너지 타겟값
        d_targets=None,    # 지속시간 타겟값
        p_control=1.0,     # 각 피처 세부조정을 위한 control 변수들
        e_control=1.0,
        d_control=1.0,
    ):
        # src_masks 생성. 데이터는 False, 패딩 부분은 True로 설정
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        # mel_lens가 주어진 경우 mel_masks 생성. 
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        # 인코딩
        output = self.encoder(texts, src_masks)

        # speaker_emb이 주어졌을 때
        if self.speaker_emb is not None:
            # 인코더 출력에 화자 임베딩을 더해서 모델이 화자 특성도 고려할 수 있게 함
            # speaker_emb(speakers): speaker 임베딩 벡터
            # .unsqueeze(1): 지정한 차원 자리에 size가 1인 빈 공간을 채워주면서 차원 확장
            # .expand(-1, max_src_len, -1): speaker 임베딩을 max_src_len에 맞춤
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # variance adaptor 거친 후
        (
            output,               # variance adaptor 거친 결과 (원래 인코더 출력 + 피치, 에너지, 지속시간 variance)
            p_predictions,        # 피치 예측값
            e_predictions,        # 에너지 예측값
            log_d_predictions,    # log 스케일된 지속시간 예측값
            d_rounded,            # 반올림된 지속시간
            mel_lens,             # 멜 스펙트로그램 길이
            mel_masks,            # 멜 스펙트로그램 마스크
        ) = self.variance_adaptor(
            output,        # 인코더 출력
            src_masks,     # src 마스크
            mel_masks,     # 멜 스펙트로그램 마스크
            max_mel_len,   # 배치 내에서 가장 긴 멜 스펙트로그램 길이
            p_targets,     # 피치 타겟 값
            e_targets,     # 에너지 타겟 값
            d_targets,     # 지속시간 타겟 값
            p_control,     # 각 피처 미세 조정을 위한 값
            e_control,
            d_control,
        )

        # variance adaptor 통과 후의 결과와 멜 스펙트로그램 마스크를 디코더에 전달
        # -> output에 디코더로 변환된 멜 스펙트로그램 저장
        output, mel_masks = self.decoder(output, mel_masks)
        # 멜 스펙트로그램 선형 변환 레이어를 거쳐 실제 멜 스펙트로그램의 차원으로 매핑해서 최종 결과 생성
        output = self.mel_linear(output)

        # 최종 output을 postnet에 통과시키고, output에 더해서 후처리 한 결과 반영 (퀄리티 상승)
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
