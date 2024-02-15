import torch
import numpy as np
from scipy.io.wavfile import write

from audio.audio_processing import griffin_lim

# audio로부터 멜스펙트로그램과 에너지 계산해서 반환
def get_mel_from_wav(audio, _stft):
    # torch.clip: 오디오 신호값을 [-1, 1]로 제한
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    # 전처리된 오디오 신호에 대한 멜스펙트로그램과 에너지 계산
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec, energy

# 멜스펙트로그램을 받아서, 원래 오디오 신호로 복원
# 멜스펙트로그램의 dynamic range 복원 & Griffin-Lim 알고리즘으로 위상 정보 없이 크기로만 오디오 복원
def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    # mel에 새로운 차원 추가해서 batch처리 가능하게끔 함
    mel = torch.stack([mel])
    # 멜스펙트로그램의 dynamic range 복원
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    # Griffin-Lim으로 복원된 스펙트로그램으로 오디오 복원함
    # phase 정보 없이 magnitude만 사용함
    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    # 복원된 오디오를 파일에 저장
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)
