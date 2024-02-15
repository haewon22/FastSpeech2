import torch
import numpy as np


class ScheduledOptim:
    # learning rate 스케쥴링을 위한 wrapper 역할을 하는 클래스
    # Vaswani et al.(2017)
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        # Adam: Adaptive Moment Estimation. 
        # 학습률을 파라미터다 적응적으로 조절하는 최적화 알고리즘. 현재 gradient와 이전 gradient의 지수 이동 평균을 이용해서 가중치 업데이트
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],  # 0.9, 0.98
            eps=train_config["optimizer"]["eps"],      # 10^(-9)
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        # Learning rate warmup
        # training초기에 모든 파라미터들이 보통 random values라서, 최종 solution에 멀리 떨어져있음
        # 이때 너무 큰 learning rate를 사용하면 numerical instability가 발생할 수 있기 때문에
        # 초기에 작은 learning rate을 사용하고, training이 안정되면 초기 learning rate으로 전환하는 방법
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        # Annealing: learning rate를 점진적으로 감소시킴 -> 안정성 증가
        # anneal step: learning rate를 감소시키기 시작하는 특정 학습 단계. 이 단계들을 기점으로 감소함
        # anneal rate: anneal step에서 얼마나 감소시킬 지 비율
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        # 현재 train 단계 저장
        self.current_step = current_step
        # 초기 learning rate 설정
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

    def step_and_update_lr(self):
        # learning rate 업데이트
        self._update_learning_rate()
        # loss function을 최적화 하도록 파라미터 업데이트
        self._optimizer.step()

    # optimizer의 gradient 초기화
    def zero_grad(self):
        # print(self.init_lr)
        # gradient를 0으로 설정
        self._optimizer.zero_grad()

    # 주어진 경로에서 optimizer의 state를 load함
    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    # 현재 train 단계에 따라서 lr 스케일 계산
    def _get_lr_scale(self):
        lr = np.min(
            [
                # 현재 단계에 -0.5제곱근을 곱해서 초기에 lr이 점진적으로 증가할 수 있게 함
                np.power(self.current_step, -0.5),
                # warm up 단계의 -1.5제곱근에 현재 단계를 곱해서 warm up 단계에서 lr이 linear하게 증가하게 함??
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        # anneal 단계 s보다 현재 단계가 더 크면 lr에 anneal_rate을 곱해서 학습률 감소
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        # 계산된 lr 리턴
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        # 초기 lr에 계산된 lr을 곱해서 현재 lr 계산
        lr = self.init_lr * self._get_lr_scale()

        # 각 파라미터 그룹에 대해 lr 계산
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
