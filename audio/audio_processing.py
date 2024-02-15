import torch
import numpy as np
import librosa.util as librosa_util
from scipy.signal import get_window


# window함수의 제곱 계산 => stft에서 window로 인해 유도된 변조 효과를 추정하는데에 사용
def window_sumsquare(
    window,             # window 함수의 종류 (window: STFT는 신호를 frame별로 잘라서 FFT를 수행하는데, 이 frame을 어떻게 자를 것인가~)
    n_frames,           # 분석할 frames의 수
    hop_length,         # 윈도우가 겹치지 않는 샘플 수 
    win_length,         # window 함수 길이 (기본값: n_fft). window 함수에 들어가는 샘플 크기. 데이터를 어느 크기로 자를 건 지
    n_fft,              # STFT에 사용되는 FFT의 크기. win_length로 잘린 음성 조각이 0으로 padding 돼서 n_fft로 크기가 맞춰짐
    dtype=np.float32,   # output의 데이터 타입
    norm=None,          # window를 normalization하기 위한 옵션
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """

    # FFT는 Time domain에 대한 정보를 나타내지 못함 그래서 STFT는 시간을 window_size 크기의 프레임별로 나눠서 FFT함
    # frame을 overlap하는 이유: frame의 양 끝쪽 신호가 자연스럽게 연결되게 하기 위함

    if win_length is None:  # window의 길이가 주어지지 않은 경우 초깃값을 n_fft로 설정
        win_length = n_fft

    # window sum-square을 담을 배열의 크기
    n = n_fft + hop_length * (n_frames - 1)
    # 누적될 sum square을 저장할 배열
    # 길이가 n, 데이터 타입이 dtype인 배열 x, 여기서 dtype은 np.float32 (배열의 element가 32비트 부동 소수점)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    # window함수 생성 (window: 함수 종류, win_length: 윈도우 함수 길이, fftbins=True: FFT에 맞게 윈도우 설정 할 건지)
    win_sq = get_window(window, win_length, fftbins=True)
    # 생성된 window 함수를 정규화. norm: 정규화 방법. 제곱: 신호값을 양수로 만들어주기 위함?
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    # pad_center(data, size): win_sq의 길이가 n_fft와 다를 때 부족한 만큼 양쪽을 0으로 채워 동일한 크기로 확장
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        # 각 프레임의 첫 번째 위치. 즉 한 프레임씩 보기 위해 연산
        sample = i * hop_length
        # 현재 프레임 ((sample : sample + n_ftt), min인 이유는 전체 n을 넘지 않기 위함)에 윈도우 함수를 적용할 부분
        # n_fft(현재 프레임에서 사용할 윈도우 길이) 만큼 사용. 맨 끝 부분에서 윈도우 함수가 잘려 사용될 때 n - sample. 이 값이 음수일 때 0 사용
        # --> win_sq[: max(0, min(n_fft, n - sample))]: 윈도우 함수 제곱 값을 현재 프레임에 적용하는 것
        # 결과: 각 프레임마다 윈도우 함수를 제곱한 결과를 x에 누적 (sum-square)
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x

# 스펙트로그램 크기로(magnitudes)부터 오디오 신호 복원
# Griffin-Lim: Mel-spectrogram으로 계산된 STFT magnitude 값만 가지고 원본 음성을 예측하는 rule-based 알고리즘
# random으로 phase를 설정하고, 미리 정해둔 횟수만큼 ISTFT와 STFT를 반복하며 적절한 phase로 수정해나감
def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """
    # 초기 위상(phase)을 임의로 두고 예측 시작함 -> 매번 결과가 다를 수 있음
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    
    # 추출된 위상을 PyTorch 텐서로 변환하고, 미분 가능(Variable)으로 만듦. 
    # --> PyTorch의 최적화 및 자동 미분 기능 사용 가능
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    # 크기와 위상만으로 ISTFT를 수행해서, 시간 영역 신호를 얻음. squeeze(1)는 불필요한 차원 제거
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):  # 미리 설정해둔 반복횟수인 n_iters 만큼 반복
        # 현재 신호에 대해 STFT를 수행하고, 새로운 angle을 얻음. _는 크기 정보를 무시한다는 의미
        _, angles = stft_fn.transform(signal)
        # 새로운 angle과 크기를 사용해서 다시 ISTFT 수행 
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
        # => 반복함으로써 신호를 점진적으로 개선함
    return signal


# 신호의 크기 차이 조정
# dynamic range: 신호에서 가장 작은 값과 큰 값의 차이

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    # 입력 값 x가 clip_val보다 작지 않게 함. 너무 작은 경우 log연산 시 방해
    # *C: compression 계수, 얼마나 압축할 건 지
    # log: 큰 값을 더 작게, 작은 값을 상대적으로 덜 작게 만들어서 dynamic range를 줄여줌
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    # exp(x): 지수함수에 x를 넣어서, 압축했던(log) dynamic range를 복원
    # /C: 조정했던 압축 정도를 되돌림
    return torch.exp(x) / C
