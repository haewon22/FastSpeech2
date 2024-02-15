import torch
import torch.nn as nn

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()

        # ("phoneme_level", "frame_level")
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        # Mean Squared Error, MSE
        self.mse_loss = nn.MSELoss()
        # Mean Absolute Error, MAE
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,       # 실제 멜 스펙트로그램 타겟
            _,                 # loss 계산에 필요 없는 값이라 _ 처리
            _,
            pitch_targets,     # 실제 피치 타겟값
            energy_targets,    # 실제 에너지 타겟값
            duration_targets,  # 실제 지속시간 타겟값
        ) = inputs[6:]
        (
            mel_predictions,          # 모델이 생성하는 멜 스펙토그램 예측값
            postnet_mel_predictions,  # postnet통과 후 멜 스펙토그램 예측값
            pitch_predictions,        # 피치 예측값
            energy_predictions,       # 에너지 예측값
            log_duration_predictions, # log 스케일 된 지속시간 예측값
            _,
            src_masks,                # 소스 마스크
            mel_masks,                # 멜 스펙토그램 마스크
            _,
            _,
        ) = predictions
        # 불리언 값 not 연산
        # 원래 패딩된 부분이 True, 데이터 부분이 False인데, 데이터 부분을 True로 (맞나)
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        # 지속시간 타겟값을 log 스케일하고, +1 더함(로그 함수에 0 들어가는거 방지용?)
        log_duration_targets = torch.log(duration_targets.float() + 1)
        # 멜 스펙토그램 타겟 길이를 유효한 데이터 길이만큼 마스킹
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        # 멜 스펙토그램 마스크 길이도 현재 멜 스펙트로그램 크기로 조절 (같은 크기로?..)
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        # loss 계산에 gradient 가 불필요해서 각 피처에 대해 requires_grad를 false로 설정
        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        # 피치가 phoneme level에서 처리되는 경우
        if self.pitch_feature_level == "phoneme_level":
            # src_masks를 사용해서 유효하지 않은 부분 마스크 처리
            # 위에서 ~연산으로 True로 바꾼 부분만 select
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        # 피치가 frame level에서 처리되는 경우
        elif self.pitch_feature_level == "frame_level":
            # mel_masks를 사용해서 유효하지 않은 부분 마스크 처리
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        # 에너지가 phoneme level에서 처리되는 경우
        if self.energy_feature_level == "phoneme_level":
            # src_masks를 사용해서 유효하지 않은 부분 마스크 처리
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        # 에너지가 frame level에서 처리되는 경우
        if self.energy_feature_level == "frame_level":
            # mel_masks를 사용해서 유효하지 않은 부분 마스크 처리
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        # 지속시간에 src_masks를 사용해서 유효하지 않은 부분 마스크 처리
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        # 멜 스펙토그램 예측값, 타겟값과, postnet을 거친 멜스펙토그램 예측값에 대해 유효하지 않은 부분 마스크 처리
        # .unsqueeze(-1): 지정한 차원 자리에 size가 1인 빈 공간을 채워주면서 차원 확장 -> 마스크를 원본 데이터 차원에 맞게 해줌
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # 멜 스펙토그램 예측값과 타겟값 사이 MAE loss 계산
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        # postnet을 통과한 멜 스펙토그램과 멜 스펙토그램 타겟값 사이 MAE loss 계산
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        # 피치 예측값과 피치 타겟값 사이 MSE loss 계산
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        # 에너지 예측값과 에너지 타겟값 사이 MSE loss 계산
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        # log 스케일링 된 지속시간 예측값과 타겟값 사이 MSE loss 계산
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # 위에서 계산된 loss를 다 더해서 총 손실을 구함
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
