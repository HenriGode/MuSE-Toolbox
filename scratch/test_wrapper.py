import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Base(nn.Module, ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'forward' in cls.__dict__:
            original_forward = cls.__dict__['forward']
            def wrapped_forward(self, x: torch.Tensor, *args, **kwargs):
                if x.dim() == 3:
                    print("Squeezed dimension detected, unsqueezing!")
                    x = x.unsqueeze(1)
                return original_forward(self, x, *args, **kwargs)
            cls.forward = wrapped_forward

class Combinator(Base):
    def forward(self, x: torch.Tensor):
        b, c, f, t = x.shape
        print(f"Shape unpacked successfully: {b}, {c}, {f}, {t}")
        return x

c = Combinator()
c.forward(torch.randn(2, 257, 100))
c(torch.randn(2, 257, 100))
