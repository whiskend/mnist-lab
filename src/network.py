# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio

        layer_dims = [784, 512, 256, 10]
        self.params = OrderedDict()

        for i in range(1, len(layer_dims)):
            input_dim = layer_dims[i - 1]
            output_dim = layer_dims[i]
            self.params[f"W{i}"] = np.random.randn(input_dim, output_dim) * np.sqrt(
                2.0 / input_dim
            )
            self.params[f"b{i}"] = np.zeros(output_dim)

            if self.use_batchnorm and i < len(layer_dims) - 1:
                self.params[f"gamma{i}"] = np.ones(output_dim)
                self.params[f"beta{i}"] = np.zeros(output_dim)

        self.layers = OrderedDict()
        hidden_layer_count = len(layer_dims) - 2

        for i in range(1, hidden_layer_count + 1):
            self.layers[f"Affine{i}"] = Affine(self.params[f"W{i}"], self.params[f"b{i}"])
            if self.use_batchnorm:
                self.layers[f"BatchNorm{i}"] = BatchNorm(
                    self.params[f"gamma{i}"], self.params[f"beta{i}"]
                )
            self.layers[f"ReLU{i}"] = ReLU()
            if self.use_dropout:
                self.layers[f"Dropout{i}"] = Dropout(self.dropout_ratio)

        output_layer_idx = hidden_layer_count + 1
        self.layers[f"Affine{output_layer_idx}"] = Affine(
            self.params[f"W{output_layer_idx}"],
            self.params[f"b{output_layer_idx}"],
        )
        self.softmax = Softmax()
        self.grads = OrderedDict(
            (key, np.zeros_like(value)) for key, value in self.params.items()
        )

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        out = x
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return self.softmax.forward(out)

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        dout = self.softmax.backward(dout)

        for name, layer in reversed(self.layers.items()):
            dout = layer.backward(dout)

            if isinstance(layer, Affine):
                idx = name.replace("Affine", "")
                self.grads[f"W{idx}"] = layer.dW
                self.grads[f"b{idx}"] = layer.db
            elif isinstance(layer, BatchNorm):
                idx = name.replace("BatchNorm", "")
                self.grads[f"gamma{idx}"] = layer.dgamma
                self.grads[f"beta{idx}"] = layer.dbeta

        return dout

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)
