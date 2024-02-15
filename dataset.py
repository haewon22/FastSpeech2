import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D

class Dataset(Dataset):
    def __init__(
        self,
        filename,            # 오디오 정보에 대한 정보가 담긴 파일 이름
        preprocess_config,   # 전처리 관련 설정이 담긴 파일
        train_config,        # train 관련 설정이 담긴 파일
        sort=False,          # 배치를 생성할 때 샘플을 정렬할 것인지 여부
        drop_last=False      # 데이터를 배치 단위로 불러왔을 때, 데이터셋 마지막 샘플이 배치 크기보다 작으면 drop할 지
    ):
        # 전처리 관련 설정에서 지정된 데이터셋 이름
        self.dataset_name = preprocess_config["dataset"]
        # 전처리된 데이터가 저장된 경로
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        # 텍스트 데이터를 clean 하기 위함
        # clean: 갖고 있는 데이터에서 노이즈 데이터 제거
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        # train에 사용될 배치 크기
        self.batch_size = train_config["optimizer"]["batch_size"]

        # process_meta 함수로 메타 데이터에서 추출한 basename, speaker 정보, 처리된 텍스트, 원본 텍스트
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        # 전처리된 파일이 저장된 경로와, speakers.json 을 join한 경로 열기
        # with: 파일 작업이 끝나면 자동으로 파일 닫힘
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            # json.load(): json데이터를 읽고, 딕셔너리로 변환
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    # 데이터셋에 포함된 샘플의 총 개수를 리턴하는 magic method
    # -> 데이터를 불러오는 데이터 로더가 데이터셋의 크기 인식 & 로딩 과정에서 반복 횟수를 정할 수 있게 함
    def __len__(self):
        return len(self.text)

    # Dataset 클래스에서 인덱스(idx)에 해당하는 샘플 불러오는 magic method
    def __getitem__(self, idx):
        # idx 인덱스에 해당하는 샘플의 정보 저장
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        # 텍스트 데이터를 cleaners 옵션을 사용해서 텍스트를 음소 시퀀스로 바꾸고, numpy 배열로 변환 
        # text_to_sequence: Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        
        # 현재 샘플에 해당하는 (멜스펙트로그램, pitch, energy, duration) 파일 경로 생성
        # np.load(".npy"): .npy파일을 불러오고, 안에 들어있는 정보가 리턴
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        # 위에서 처리한 정보로 현재 샘플의 딕셔너리 생성
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    # 메타 데이터 파일에서부터 basename, speaker 정보, 처리된 텍스트, 원본 텍스트 정보를 추출하는 함수
    def process_meta(self, filename):
        # 전처리된 파일이 저장된 경로와, filename을 join한 경로를 읽기 전용으로 열기
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []

            # f.readlines(): 파일의 모든 줄을 리스트로 변환
            for line in f.readlines():
                # \n을 제거하고, |를 기준으로 문자열 분리
                n, s, t, r = line.strip("\n").split("|")
                # 메타데이터에서 정보 추출
                name.append(n)       # basename
                speaker.append(s)    # speaker 정보
                text.append(t)       # 처리된 텍스트
                raw_text.append(r)   # 원본 텍스트
            return name, speaker, text, raw_text

    # 데이터셋 샘플을 학습이나 평가에 필요한 형태로 재가공
    def reprocess(self, data, idxs):
        # data에서 정보별로 필요한 정보 추출
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        # 텍스트 길이
        text_lens = np.array([text.shape[0] for text in texts])
        # 멜 스펙스토그램 길이
        mel_lens = np.array([mel.shape[0] for mel in mels])

        # 데이터 패딩 -> 배치에서 데이터의 크기를 맞추는 작업
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,               # 샘플 ID
            raw_texts,         # 원본 텍스트
            speakers,          # speaker ID
            texts,             # 패딩 처리한 텍스트
            text_lens,         # 텍스트 길이 numpy 배열
            max(text_lens),    # 최대 텍스트 길이 
            mels,              # 패딩 처리한 멜 스펙트로그램
            mel_lens,          # 멜 스펙트로그램 길이 numpy 배열
            max(mel_lens),     # 멜 스펙트로그램 최대 길이
            pitches,           # 패딩 처리된 pitch
            energies,          # 패딩 처리된 에너지
            durations,         # 패딩 처리된 지속시간
        )

    # 데이터 샘플들이 다 다른 크기나 형태일 수 있으니까 모델이 처리할 수 있게 일관된 형태의 배치로 가공하는 역할
    # 파이토치의 데이터로더가 배치를 생성할 때 호출됨
    def collate_fn(self, data):
        # 데이터셋 샘플의 개수
        data_size = len(data)

        # sort가 True라면 샘플 텍스트 길이로 정렬
        if self.sort:
            # 샘플 텍스트 길이 numpy 배열
            len_arr = np.array([d["text"].shape[0] for d in data])
            # 샘플 텍스트 길이 내림차순 정렬
            idx_arr = np.argsort(-len_arr)
        # sort 하지 않을 거라면 원래 순서대로 처리
        else:
            # np.arange(data_size): 0 ~ data_size-1
            idx_arr = np.arange(data_size)

        # 마지막 샘플이 배치 크기보다 작을 때 tail로 분리
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        # tail을 제외한 배치 크기에 맞는 배열로 idx_arr 업데이트
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        # list로 변환
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        # drop_last가 false고, tail이 존재할 때
        if not self.drop_last and len(tail) > 0:
            # tail을 idx_arr리스트에 추가
            idx_arr += [tail.tolist()]

        output = list()
        # 위에서 정해진 처리순서대로 data를 reprocess
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        # 모든 배치 데이터가 담긴 리스트
        return output


class TextDataset(Dataset):
    def __init__(
            self, 
            filepath,          # 메타 데이터 파일 경로
            preprocess_config  # 전처리 관련 설정이 담긴 파일
            ):
        # 텍스트 클리닝에 사용될 설정 리스트
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        # process_meta 함수로 메타 데이터에서 추출한 basename, speaker 정보, 처리된 텍스트, 원본 텍스트
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        # 전처리된 파일이 저장된 경로와, speakers.json 을 join한 경로 열기
        # with: 파일 작업이 끝나면 자동으로 파일 닫힘
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            # json.load(): json데이터를 읽고, 딕셔너리로 변환
            self.speaker_map = json.load(f)

    # 데이터셋에 포함된 샘플의 총 개수를 리턴하는 magic method
    # -> 데이터를 불러오는 데이터 로더가 데이터셋의 크기 인식 & 로딩 과정에서 반복 횟수를 정할 수 있게 함
    def __len__(self):
        return len(self.text)

    # Dataset 클래스에서 인덱스(idx)에 해당하는 샘플 불러오는 magic method
    def __getitem__(self, idx):
        # idx 인덱스에 해당하는 샘플의 정보 저장
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        # 텍스트 데이터를 cleaners 옵션을 사용해서 음소 시퀀스로 바꾸고, numpy 배열로 변환 
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        # filename 경로를 읽기 전용으로 열기
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            # f.readlines(): 파일의 모든 줄을 리스트로 변환
            for line in f.readlines():
                # \n을 제거하고, |를 기준으로 문자열 분리
                n, s, t, r = line.strip("\n").split("|")
                # 메타데이터에서 정보 추출
                name.append(n)       # basename
                speaker.append(s)    # speaker 정보
                text.append(t)       # 처리된 텍스트
                raw_text.append(r)   # 원본 텍스트
            return name, speaker, text, raw_text

    # 파이토치의 데이터로더가 배치를 생성할 때 호출됨
    def collate_fn(self, data):
        ids = [d[0] for d in data]                                # 샘플 id
        speakers = np.array([d[1] for d in data])                 # speaker id를 numpy 배열로 저장
        texts = [d[2] for d in data]                              # 처리된 텍스트
        raw_texts = [d[3] for d in data]                          # 원본 텍스트
        text_lens = np.array([text.shape[0] for text in texts])   # 텍스트 길이를 numpy 배열로 저장

        # pad_1D로 길이가 다른 텍스트를 패딩 -> 모든 텍스트 시퀀스 길이 맞춤
        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    # 데이터셋
    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    # 배치처리
    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
