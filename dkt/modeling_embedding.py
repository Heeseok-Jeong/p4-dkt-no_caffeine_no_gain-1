from typing import Optional

import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter

from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import init

class ContinuousEmbedding(Module):

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    max_norm: float
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 num_points=20, minval=0.0, maxval=1.0, window_type='hann',
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, window_size: int = 8, _weight: Optional[Tensor] = None) -> None:
        super(ContinuousEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.minval = minval
        self.maxval = maxval
        self.window_type = window_type
        self.num_points = num_points
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.window_size = window_size
        self.points = torch.range(0, self.num_points-1, dtype=torch.float32)
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

        if self.window_type == 'hann':
            self.window_func = self._hann_window
        elif self.window_type == 'triangular':
            self.window_func = self._triangle_window
        else:
            self.window_func = self._rect_window
        self.sparse = sparse

    def _rect_window(self, x, window_size=3):
        w_2 = window_size / 2
        return (torch.sign(x + w_2) - torch.sign(x - w_2)) / 2

    def _triangle_window(self, x, window_size=3):
        w_2 = window_size / 2
        return (torch.abs(x + w_2) + torch.abs(x - w_2) - 2 * torch.abs(x)) / window_size

    def _hann_window(self, x, window_size=3):
        y = torch.cos(np.math.pi * x / window_size)
        y = y * y * self._rect_window(x, window_size=window_size)
        return y

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:

        w = input.unsqueeze(-1) - self.points.to("cuda")
        w = self.window_func(w, window_size = self.window_size)

        return torch.matmul(w, F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse))

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):

        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding