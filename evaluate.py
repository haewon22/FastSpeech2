import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, step, configs, logger=None, vocoder=None):
    # 각 config 저장
    preprocess_config, model_config, train_config = configs

    # Get dataset - 데이터셋 준비
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    # 배치 사이즈 설정
    batch_size = train_config["optimizer"]["batch_size"]
    # 데이터 로더 설정
    loader = DataLoader(
        # 로딩할 데이터넷
        dataset,
        # 한 번에 로딩할 배치 크기
        batch_size=batch_size,
        # 데이터를 랜덤으로 섞을지 여부 -> overfitting 방지
        shuffle=False,
        # 데이터 로더가 데이터셋에서 로딩할 때, 샘플들을 모델에 맞는 적절한 형태의 배치로 변환
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    # loss function 초기화
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    # 각 손실 총합을 저장할 리스트 생성
    loss_sums = [0 for _ in range(6)]
    
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            # gradient 계산 비활성화
            with torch.no_grad():
                # Forward
                # 모델로 예측값 계산
                output = model(*(batch[2:]))

                # Calculate Loss
                losses = Loss(batch, output)

                # 각 loss 총합 계산
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    # loss 평균
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    # logger가 주어졌을 때
    if logger is not None:
        # fig: plot_mel()함수로 멜스펙트로그램 시각화한 결과
        # wav_reconstruction: 타겟 멜 스펙트로그램으로 원본과 유사하게 vocoder로 복원된 오디오 신호
        # wav_prediction: 예측된 mel_prediction으로 합성된 원본과 유사하게 vocoder로 복원된 오디오 신호
        # tag: basename
        # (tools.py 참고)
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        # 전처리 설정에서 sampling rate 가져옴
        # sampling_rate: 초당 샘플 수
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        # wav_reconstruction를 tag와 log
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        # wav_prediction를 tag와 log
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)
