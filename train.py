import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    print("Prepare training ...")

    # 각 config 저장
    preprocess_config, model_config, train_config = configs

    # Get dataset - 데이터셋 준비
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    # 배치 사이즈 설정
    batch_size = train_config["optimizer"]["batch_size"]
    # 한 번에 처리할 그룹의 데이터 수 (32개 샘플이 있으면, 4그룹, 8배치로 나누어서 처리함)
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    # 배치 크기와 그룹 크기의 곱이 데이터셋보다 작아야 함
    assert batch_size * group_size < len(dataset)
    # 데이터 로더 설정
    loader = DataLoader(
        # 로딩할 데이터셋
        dataset,
        # 한 번에 로딩할 배치 크기
        batch_size=batch_size * group_size,
        # 데이터를 랜덤으로 섞을지 여부 -> overfitting 방지
        shuffle=True,
        # 데이터 로더가 데이터셋에서 로딩할 때, 샘플들을 모델에 맞는 적절한 형태의 배치로 변환
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    # get_model()로 모델과 optimizer 가져옴
    model, optimizer = get_model(args, configs, device, train=True)
    # GPU에서 계산할 때 병렬로 처리할 수 있게 하는 듯
    model = nn.DataParallel(model)
    # 매개변수 개수
    num_param = get_param_num(model)
    # loss function 초기화
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    # step 업데이트를 위한 변수
    step = args.restore_step + 1
    epoch = 1
    # gradient accumulation step
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    # 모델의 총 훈련 단계
    total_step = train_config["step"]["total_step"]
    # 로그 기록을 위한 단계
    log_step = train_config["step"]["log_step"]
    # 저장 단계 (체크포인트)
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    # 진행 상태 표시해주는 바
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    # 훈련이 완료될 떄까지 
    while True:
        # 에포크 상황 표시를 위한 진행 상태 표시바
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        # 로더에서 배치를 차례대로 가져옴
        for batchs in loader:
            # 가져온 배치 순회
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                # 모델로 예측값 계산
                output = model(*(batch[2:]))

                # Calculate Loss
                # 예측된 값 output과 batch로 loss 계산
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                # 각 grad_acc_step 단계 처리
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                # 각 log_step 단계 처리 -> 특정 스탭에서 로그 기록하기 위함
                # 현재 스탭이 log를 기록해야 하는 스탭일 때
                if step % log_step == 0:
                    # 각 loss 값들을 리스트로 저장 (.item(): 파이썬 숫자로 값 변경)
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    # log.txt 파일을 열어서 스탭과, Loss 정보 기록
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                # 현재가 synthesize 단계일 때
                if step % synth_step == 0:
                    # fig: plot_mel()함수로 멜스펙트로그램 시각화한 결과
                    # wav_reconstruction: 타겟 멜 스펙트로그램으로 원본과 유사하게 vocoder로 복원된 오디오 신호
                    # wav_prediction: 예측된 mel_prediction으로 합성된 원본과 유사하게 vocoder로 복원된 오디오 신호
                    # tag: basename
                    # (tools.py 참고)
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,              # 현재 처리 중인 배치
                        output,             # 현재 배치에 대한 예측 결과
                        vocoder,            # 멜 스펙트로그램 -> 오디오 신호
                        model_config,       # 모델 설정
                        preprocess_config,  # 전처리 설정
                    )
                    # 시각화된 fig를 tag와 log
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    # 전처리 설정에서 sampling rate 가져옴
                    # sampling_rate: 초당 샘플 수
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    # wav_reconstruction를 tag와 log
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    # wav_prediction를 tag와 log
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                # 현재 스탭이 validation 스탭이라면
                if step % val_step == 0:
                    # 모델을 evaluation(평가) 모드로 전환 (학습시에 필요한 옵션 비활성화함)
                    model.eval()
                    # evaluate() 함수로 모델 평가
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    # 모델을 train모드로 전환 (비활성화 했던 옵션들 다시 활성화)
                    model.train()

                # 현재 스탭이 save step이라면 (체크 포인트)
                if step % save_step == 0:
                    # 현재 스탭에서 모델과 optimizer의 state를 저장
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                # 현재 스탭이 설정된 전체 스탭이라면 (마지막 스탭) 종료
                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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

    main(args, configs)
