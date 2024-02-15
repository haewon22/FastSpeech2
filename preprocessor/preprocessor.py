import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

class Preprocessor:
    def __init__(self, config):
        self.config = config
        # raw 오디오 파일이 저장된 디렉토리 path
        self.in_dir = config["path"]["raw_path"]
        # preprocessing한 오디오가 저장될 dir path
        self.out_dir = config["path"]["preprocessed_path"]
        # validation dataset size
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        # 한 프레임에서 다음 프레임까지의 샘플 수
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        # pitch feature를 계산하는게 phoneme_level 또는 frame_level인지 확인
        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        # energy feature를 계산하는게 phoneme_level 또는 frame_level인지 확인
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        # pitch phonme 평균을 phoneme_level로 계산할 건지
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        # energy phonme 평균을 phoneme_level로 계산할 건지
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        
        # pitch를 정규화 할 건지
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        # energy를 정규화 할 건지
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    # in_dir로 오디오를 읽어 preprocessing, 결과를 out_dir에 저장
    def build_from_path(self):
        # 출력 디렉토리(mel, pitch, energy, duration) 생성
        # os.path.join이 out_dir/mel등의 이름의 하위 디렉토리 경로 생성
        # os.makedirs 경로에 해당하는 디렉토리 생성
        # exist_ok: True라면 해당 경로에 이미 디렉토리가 존재하더라도 에러 발생x
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        # 전처리 결과를 저장할 리스트
        out = list()
        # 처리된 프레임 수
        n_frames = 0
        # 정규화를 위한 인스턴스
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}  # 딕셔너리
        # in_dir 경로에 있는 모든 디렉토리(speaker) 순회
        # os.listdir: 지정 경로의 디렉토리 내의 모든 파일 이름을 리스트로 리턴
        # tqdm: 진행상황 바 모듈
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            # speaker 딕셔너리에 화자 이름(speaker)를 key로, 그 화자의 index i를 value로 저장
            speakers[speaker] = i
            # 현재 speaker에 대한 디렉토리 내에 있는 모든 파일 순회
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                # 오디오 파일(.wav)만 처리하기 위해 확인
                if ".wav" not in wav_name:
                    continue

                # . 기준으로 파일 이름 분리하고, [0]번째 요소인 파일 이름을 basename에 저장
                basename = wav_name.split(".")[0]
                # "out_dir/TextFrid/speaker/basename.TextGrid" 경로
                # TextGrid 파일: 오디오 파일에 대한 주석 파일이라고 생각하면 됨
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                # 위에서 만든 tg_path 경로가 존재한다면
                if os.path.exists(tg_path):
                    # process_utterance: (tuple[str, ndarray[Any, dtype], ndarray[Any, dtype], Any] | None)반환
                    # 텍스트 정보(info), 피치 정보 numpy array, 에너지 정보 numpy array, any정보를 튜플로 리턴
                    ret = self.process_utterance(speaker, basename)
                    # process_utterance 리턴값이 None인 경우 처리할 정보가 없으므로 continue
                    if ret is None:
                        continue
                    # 튜플로 받아온 정보 각 변수에 저장
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)

                # 추출된 피치값이 있는 경우 피치 데이터로 StandardScaler의 평균과 표준편차 계산하고 업데이트
                # pitch.reshape(-1, 1): 피치 데이터를 2차원으로 변환 -> StandartScalar가 2차원을 기대하기 때문
                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                # 추출된 에너지값이 있는 경우
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                # 처리된 프레임수를 업데이트
                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        # pitch_normalization값이 True라면
        if self.pitch_normalization:
            # StandScalar의 객체 pitch_scaler에서 계산된 평균을 pitch_mean에 저장
            pitch_mean = pitch_scaler.mean_[0]
            # 동일하게 계산된 표준편차를 pitch_std에 저장
            pitch_std = pitch_scaler.scale_[0]
        # pitch_normalization값이 False라면
        else:
            # A numerical trick to avoid normalization...
            # 평균을 0, 표준편차를 1로 설정해 정규화하지 않도록 설정
            pitch_mean = 0
            pitch_std = 1
        # energy normalization값이 True라면
        if self.energy_normalization:
            # energy_scaler에서 계산된 평균을 energy_mean에 저장
            energy_mean = energy_scaler.mean_[0]
            # 계산된 표쥰편차를 energy_std에 저장
            energy_std = energy_scaler.scale_[0]
        # 에너지 정규화를 하지 않는다면 (energy_normalization이 False)
        else:
            # 평균을 0, 표준편차를 1로 설정해 정규화하지 않도록 설정
            energy_mean = 0
            energy_std = 1

        # normalize 함수를 이용해서 피치 데이터 정규화 후 min, max 값 저장
        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        # normalize 함수를 이용해서 에너지 데이터 정규화 후 min, max 값 저장
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        # speakers.json을 쓰기 전용으로 open
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            # speakers 딕셔너리를 json 형식으로 변환해서 파일에 쓰기
            f.write(json.dumps(speakers))

        # stats.json을 쓰기 전용으로 open
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            # pitch와 energy에 대한 정규화 정보를 json 형식으로 stats에 저장
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            # stats를 stats.json 파일에 write
            f.write(json.dumps(stats))

        # 전체 오디오 시간
        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        # 처리된 데이터를 랜덤으로 섞음 -> 모델 학습 시 일반화 성능 향상
        random.shuffle(out)
        # None값 제거
        out = [r for r in out if r is not None]

        # Write metadata
        # out_dir의 train.txt파일을 쓰기 전용으로 열기
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            # 훈련 데이터 슬라이싱
            for m in out[self.val_size :]:
                f.write(m + "\n")
        # out_dir의 val.txt를 쓰기 전용으로 열기
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            # validation 데이터 슬라이싱
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    # speaker의 utterance(발화) 처리 함수
    def process_utterance(self, speaker, basename):
        # "in_dir/speaker/basename.wav" 형식의 경로를 wav_path에 저장
        # 위 .wav 파일은 화자의 발화에 해당하는 오디오 파일
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        # "in_dir/speaker/basename.lab" 형식의 경로를 text_path에 저장
        # 위 .lab 파일은 같은 발화에 대한 text 레이블이 저장된 파일
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        # "out_dir/TextGrid/speaker/basename.TextFrid" 형식의 경로를 tg_path에 저장
        # .TextGrid 파일은 오디오의 시간 정렬 정보를 포함
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        # tgt.io.read_textgrid: Praat의 TextGrid 파일을 읽고 TextGrid 객체를 리턴
        textgrid = tgt.io.read_textgrid(tg_path)

        # get_alignment 함수에 tier인자로 textgrid.get_tier_by_name("phones") 사용하여 해당 tier의 feature 추출
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )

        # 음소의 시퀀스를 text로 표현하기 위해 공백으로 구분하여 한 문자열로 join
        # 대충 "{ a b c d }"
        text = "{" + " ".join(phone) + "}"
        # 시작 시점이 종료 시점보다 크거나 같으면 None을 리턴
        if start >= end:
            return None

        # Read and trim wav files
        # wav_path 에 있는 오디오 파일을 load하고, wav와 sampling_rate을 반환
        wav, _ = librosa.load(wav_path)
        # [ 시작 시점 샘플링 인덱스 : 종료 시점 샘플링 인덱스 ] 로 슬라이싱 후, float32 데이터 타입으로 변환
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        # text_path에 해당하는 텍스트 정보가 담긴 파일을 읽기 전용으로 열기
        # text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        with open(text_path, "r") as f:
            # readline()으로 파일의 첫 번째 줄을 읽고 \n 제거
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        # pw.dio(): PyWorld 라이브러리의 dio 
        # dio: (Distributed Inline-filter Operation 알고리즘으로 피치와 피치가 추정된 각 프레임의 중심 시간(timestamp) 리턴)
        # input: (분석하려는 numpy배열, 오디오 샘플링 rate, 분석하려는 프레임 길이를 밀리초 단위로 지정한 값)
        # 리턴 값: (f0, 각 f0에 해당하는 시간 배열)
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        # dio를 통해 계산된 pitch와 t의 정확도 개선을 위해 stonemask 사용
        # stonemask: 초기 피치 추정값 주변에서 미세 조정을 통해 실제 음성 신호랑 가장 비슷한 값 찾아냄
        # input: (분석하려는 numpy배열, f0, 각 f0에 해당하는 시간 배열, sampling_rate)
        # 리턴값: 보정된 피치
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        # 발화 지속시간 합으로 슬라이싱
        pitch = pitch[: sum(duration)]
        # 유효한 피치값이 없거나 하나라면 그냥 None 리턴
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        # get_mel_from_wav: audio로부터 멜스펙트로그램과 에너지 계산해서 반환하는 함수
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        # 계산된 멜 스펙트로그램에서 지속시간에 해당하는 부분만 슬라이싱
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        # 계산된 에너지에서 지속시간에 해당하는 부분만 슬라이싱
        energy = energy[: sum(duration)]

        # 피치 평균 계산
        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            # pitch값이 0이 아닌 모든 인덱스를 nonzero_ids에 저장
            nonzero_ids = np.where(pitch != 0)[0]
            # interp1d: scipy.interpolate 함수. linear interpolation을 생성함. 샘플과 샘플 사이 공백 채워주는 역할
            # interp_fn은 nonzero_ids에서 유효한 pitch값을 가진 포인트를 기준으로 linear interpolation을 수행해서 무음 구간의 pitch를 추정
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            # 모든 pitch값에 대한 linear interpolate
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                # 지속시간이 0보다 긴 경우
                if d > 0:
                    # 해당 지속시간 구간에 대한 피치값 평균
                    pitch[i] = np.mean(pitch[pos : pos + d])
                # 지속시간이 0인 경우 해당 음소가 무음이므로 피치값을 0으로 설정
                else:
                    pitch[i] = 0
                pos += d
            # 평균 계산된 피치 배열 길이에 맞게 슬라이싱
            pitch = pitch[: len(duration)]

        # 에너지 평균 계산
        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                # 지속시간이 0보다 긴 경우
                if d > 0:
                    # 해당 지속시간 구간에 대한 에너지값 평균
                    energy[i] = np.mean(energy[pos : pos + d])
                # 지속시간이 0인 경우 해당 음소가 무음이므로 에너지값을 0으로 설정
                else:
                    energy[i] = 0
                pos += d
            # 평균 계산된 에너지 배열 길이에 맞게 슬라이싱
            energy = energy[: len(duration)]

        # Save files
        # "speaker-duration-basename.npy" 형식의 파일 경로에 지속시간 저장
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        # "speaker-pitch-basename.npy" 형식의 경로에 피치 데이터 파일 저장
        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        # "speaker-energy-basename.npy" 형식의 파일 경로에 에너지 데이터 파일 저장
        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        # "speaker-mel-basename.npy" 형식의 경로에 멜 스펙트로그램 데이터 파일 저장
        # .T: transpose. 멜 스펙트로그램의 차원 계산
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            # remove_outlier 함수를 통해 데이터 패턴에서 크게 벗어나는 outlier들을 제거
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            # 멜스펙트로그램의 두 번째 차원. 프레임 수를 나타냄.(= 처리된 오디오 파일 길이)
            mel_spectrogram.shape[1],
        )

    # 음소 alignment, 지속시간, 발화 시작 지점, 끝지점 구함
    def get_alignment(self, tier):

        # silence에 대한 음소 리스트
        # sil, sp, spn을 silence 음소로 판단
        sil_phones = ["sil", "sp", "spn"]

        phones = []      # 발화에서 인식된 음소를 저장
        durations = []   # 지속 시간
        start_time = 0   # 발화의 시작 시간
        end_time = 0     # 종료 시간
        end_idx = 0      # 마지막 음소의 index

        # tier._objects: TextGrid tier의 모든 요소
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:           # 발화의 시작일 때
                if p in sil_phones:    # text에 silence 음소가 있다면
                    continue           # 건너뜀
                else:                  # text에 silence 음소가 없다면
                    start_time = s     # 발화의 시작을 현재 start_time으로 설정

            # silence 음소가 아니라면
            if p not in sil_phones:
                # For ordinary phones
                # p를 phones에 추가하고, 종료 시간과 종료 인덱스를 업데이트
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            # 각 음소 지속시간 계산
            # sampling_rate: 초당 샘플 수 -> (e or s)*sampling_rate: 해당 시점 샘플 인덱스
            # (e*sampling_rate) / hop_length: 해당 샘플이 몇 번째 STFT 프레임인가
            # 따라서 종료 시점 프레임 인덱스에서 시작 시점 프레임 인덱스를 빼면 지속시간
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        # 발화의 종료 시점까지만 업데이트 해서 이후의 silence를 제거
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    # 데이터 패턴에서 크게 벗어나는 outlier들을 제거
    def remove_outlier(self, values):
        # input된 values를 numpy 배열로 전환
        values = np.array(values)
        # 데이터를 4개로 나눔 (사분위수, 0, 25, 75, 100으로 나눔)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        # (p75 - p25) 사이 범위에서 1.5배를 벗어나는 값을 거르기 위한 상한과 하한
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        # 상한, 하한 값을 벗어나는 outlier가 아니라면 normal_indices에 저장
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    # in_dir내의 모든 파일에 정규화 후, 최소값과 최대값 리턴
    # 사용: pitch_min, pitch_max = self.normalize(os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std)
    def normalize(self, in_dir, mean, std):
        # np.finfo(np.float64).min: float64 데이터 타입에서 표현할 수 있는 가장 작은 값
        max_value = np.finfo(np.float64).min
        # np.finfo(np.float64).max: float64 데이터 타입에서 표현할 수 있는 가장 큰 값
        min_value = np.finfo(np.float64).max
        # os.listdir: 지정 경로의 디렉토리 내의 모든 파일 이름을 리스트로 리턴
        for filename in os.listdir(in_dir):
            # "in_dir/filename": 전체 파일 경로 생성
            filename = os.path.join(in_dir, filename)
            # np.load(filename): filename 중 numpy 배열을 저장하는 파일을 load
            # numpy 배열에서 평균을 빼고, 표준편차로 나누어 정규화 (평균을 0, 표준편차를 1로)
            values = (np.load(filename) - mean) / std
            # 계산된 value값을 같은 이름의 파일 (filename)에 다시 저장
            np.save(filename, values)

            # 정규화된 값 중 max와 min
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
