import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from audio.audio_processing import (
    dynamic_range_compression,
    dynamic_range_decompression,
    window_sumsquare,
)

# Short-Time-Fourier Transform(STFT)
# FFT는 음성 신호 전체에 대해서 주파수 성분을 분석하기 때문에 시간 영역을 다루지 못함
# 그래서 일정 구간(window size)로 나누고, 각 구간에 대해서 FFT를 수행해서 spectrogram 형태로 주파수 성분 분석
class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length    # 각 프레임의 샘플 수
        self.hop_length = hop_length          # 한 프레임에서 다음 프레임까지의 샘플 수
        self.win_length = win_length          # 윈도우 함수 길이
        self.window = window                  # 윈도우 함수 종류. 여기서 기본값은 hanning
        self.forward_transform = None         # 

        # FFT basis 벡터 생성.
        # scale은 필터 길이와 hop길이의 비율 -> inverse 변환할 때 스케일링에 사용 됨
        scale = self.filter_length / self.hop_length
        # FFT의 basis vector
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        
        # FFT 결과의 중요한 부분만 사용하기 위함. FFT는 대칭이기 때문에 반만 봄
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        # STFT 분석을 위한 basis 벡터
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        # ISTFT 분석을 위한 basis 벡터
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        # 윈도우 함수를 STFT와 역변환에 사용되는 basis 벡터에 적용
        # 윈도우 함수는 FFT를 적용하기 전에 신호에 곱해서, 신호의 시작과 끝에서 발생할 수 있는 급격한 변화를 완화함 (smoothing)
        # & 신호의 특정 부분을 강조해서 정확도 향상
        if window is not None:  # 윈도우를 사용할 경우
            # 필터 길이가 윈도우 크기보다 크거나 같아야 함 -> 윈도우는 신호의 일부를 잘라내는 역할이기 때문
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            # 윈도우 함수 생성. fftbins=True면 FFT에 맞게 윈도우 생성
            fft_window = get_window(window, win_length, fftbins=True)
            # 윈도우 함수를 필터와 같은 길이로 맞추고 (양 끝 0으로 채움), 중앙에 위치하도록 패딩함. -> 윈도우 함수가 전체 신호에 적용될 수 있음
            fft_window = pad_center(fft_window, filter_length)
            # numpy배열인 fft_window를 Pytorch의 tensor로 변환하고, 데이터 타입을 float으로 변환
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases: basis 벡터에 윈도우 함수 적용
            # 윈도우 함수를 STFT에 사용할 basis 벡터에 적용 -> 각 basis 벡터에 윈도우 함수가 곱해지면 신호의 특정 부분이 강조됨
            forward_basis *= fft_window
            # 위도우 함수를 ISTFT에 사용할 basis 벡터에 적용 -> Inverse 과정에도 윈도우 함수 영향 주기
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())


    # STFT 과정
    def transform(self, input_data):

        # 배치의 크기: 동시에 처리할 오디오 신호의 수
        num_batches = input_data.size(0)
        # 오디오 샘플 수
        num_samples = input_data.size(1)
        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        # 1D conv를 적용하기 위해, input_data의 차원을 (배치크기 * 1 * 샘플 수) 로 변환
        input_data = input_data.view(num_batches, 1, num_samples)
        # F.pad()로 패딩. filer_length/2만큼의 패딩이 양 끝에 적용됨
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        # 1D conv를 사용해서 STFT 수행
        # forward_basis로 신호의 주파수 성분 추출
        # stride=self.hop_length: conv 적용할 때 stride를 홉 길이로 설정함.
        # 결과가 forward_transform에 저장
        forward_transform = F.conv1d(
            input_data.cuda(),
            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        # FFT는 대칭므로, 절반은 중복이니까 잘라냄
        cutoff = int((self.filter_length / 2) + 1)
        # conv 결과에서 실수 부분
        real_part = forward_transform[:, :cutoff, :]
        # conv 결과에서 허수 부분
        imag_part = forward_transform[:, cutoff:, :]

        # 입력 오디오 신호에 대한 각 주파수 크기. 실수부분^2 + 허수부분^2
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        # 각 주파수 성분의 위상. atan2는 허수부분과 실수 부분의 비율 구해줌
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    # ISTFT 과정: magnitude와 phase 정보를 사용해서 시간 영역으로 역변환. STFT의 결과를 다시 오디오 신호로 변환함
    def inverse(self, magnitude, phase):
        # magnitude와 phase를 재결합
        # 실수부: magnitude * cos(phase) / 허수부: magnitude * sin(phase)
        # 실수부와 허수부를 concatenate해서 복소수 형태로 만듦
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        # F.conv_transpose1d: 1D transposed convolution. 주파수 -> 시간
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,   # 위에서 계산한 복소수
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        # 윈도우 함수 영향 제거  
        # -> STFT할 때 window적용으로 겹친 신호의 에너지가 실제보다 높을 수도 있는 등 그런 window의 영향을 제거해야 함
        # 윈도우 함수가 주어졌다면
        if self.window is not None:
            # 윈도우 함수의 제곱 합 계산 (audio_processing.py 참고)
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            # window sum값이 tiny(window_sum)보다 큰 값들, 즉 window함수의 영향이 거의 없었던 부분을 제외하기 위함
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            # window_sum 배열을 numpy에서 tensor로 변환
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            
            # 1D conv로 얻은 시간 영역 신호를
            # 윈도우의 영향이 있었던 시점들의(approx_nonzero_indeces) window_sum으로 나눠줌으로써 증가된 에너지를 다시 되돌림
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]
            # scale by hop ratio
            # STFT 과정에서 프레임 간에 overlap의 영향을 보정
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    # STFT 결과로 얻은 magnitude와 phase를 사용해서 시간 영역으로 역변환
    # 신호의 특성 분석이나, 변조 하고 나서 원래 신호로 되돌리고 싶을 때 사용함
    def forward(self, input_data):
        # 입력 데이터에 STFT수행하는 transform()가 오디오를 주파수로 변환하고, magnitude와 phase 계산
        self.magnitude, self.phase = self.transform(input_data)
        # 계산된 magnitude와 phase로 ISTFT 수행: 주파수 데이터를 다시 시간 영역으로 변환해서 원래의 신호로 되돌림 (근사하게)
        reconstruction = self.inverse(self.magnitude, self.phase)
        # 재구성된 신호 리턴
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,   # FFT를 적용할 떄 사용되는 필터 크기
        hop_length,      # 한 프레임에서 다음 프레임까지의 샘플 수
        win_length,      # 윈도우 함수 길이
        n_mel_channels,  # 생성될 멜스펙트로그램의 멜 필터 뱅크 채널(mel scale에 따라 주파수 대역을 나눈 각각의 필터)의 수
        sampling_rate,   # 초당 샘플 수
        mel_fmin,        # 멜 스펙트로그램 계산할 때 사용되는 주파수 범위 최소값
        mel_fmax,        # 멜 스펙트로그램 계산할 때 사용되는 주파수 범위 최대값
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        # STFT 수행할 객체 생성
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        # 멜 필터 뱅크 생성 (주파수 영역을 멜 스케일로 매핑하는 필터 집합)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)


    # 스펙트럼의 dynamic range를 압축
    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    # 압축된 스펙트럼의 dynamic range를 복원
    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    # 입력된 오디오 신호의 batch에 대해 멜스펙트로그램 계산
    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        # y: 파이토치 텐서. 범위는 [-1. 1]인데, 오디오 신호가 일반적으로 이 범위에서 정규화됨
        # assert는 조건 확인. 범위를 벗어나면 에러 남
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        # 입력 신호 y에 대해 STFT. 
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        # 멜 스펙트로그램 계산
        # mel_basis: 멜 필터 뱅크
        # mel_basis와 STFT 결과 나온 magnitude 행렬 곱을 하면 멜 스펙트로그램 생성됨 -> 주파수 영역 데이터를 멜 스케일로
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        # 멜 스펙트로그램의 dynamic range를 압축함
        mel_output = self.spectral_normalize(mel_output)
        # 에너지 계산. magnitude에 대해 L2 norm(torch.norm은 L2가 기본) 계산하면 각 프레임의 에너지
        # L2 norm = sqrt(Σ|xᵢ|²)
        energy = torch.norm(magnitudes, dim=1)

        # 멜스펙트로그램, 에너지 반환
        return mel_output, energy
