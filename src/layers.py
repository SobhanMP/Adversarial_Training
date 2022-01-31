import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

""" Standard scaler as a layer

Shifts the average to zero and scales the standard deviation to one.
"""
class StandardScalerLayer(nn.Module):
    """
    data: function that returns an iterator
    keep_axis: iterable for dims to keep
        defaults keep second argument
    """

    def __init__(self, data, keep_dims=[1]):
        super(StandardScalerLayer, self).__init__()

        self.keep_dims = keep_dims

        c = Collectinator(torch.zeros(3))
        for inputs in data():
            inputs_mean = mean_keepdim(inputs, keep_dims)
            c.add(inputs_mean, inputs.size(0))
        mean = c.mean

        c = Collectinator(torch.zeros(3))
        for inputs in data():
            m = inputs.size(0)
            Δ = inputs - mean[None, :, None, None]
            d = mean_keepdim(Δ * Δ, keep_dims)
            c.add(d, m)
        var = c.mean

        std = torch.pow(var, 0.5)

        # so that optimizers don't change them
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.reshape(x, self.mean)) / self.reshape(x, self.std)

    def reshape(self, x, y):
        for i in range(len(x.shape)):
            if i not in self.keep_dims:
                y = torch.unsqueeze(y, i)
        return y


''' Adversarial for free Layer

This layer adds the noise as described in the paper 
Adversarial training for free. 

Attributes
----------
e: usually a float
    ensures that the noise $||noise||_\infty \leq e$
min, max: usually a float
    clamp the output so that the adversarial example stays valid
    set the $-\infty$ and $\infty$ to disable it
auto_zer_grad: bool
    reset the gradient of the noise every time forward is called
    IF YOU ARE ACCUMULATING GRADIENT, you need to disable it and 
    clear the gradient manually

Usage notes
-----------

- call .step() after calculating the gradient to update the noise
- if the size of the input changes, the buffer is dropped
- the noise is stored as a buffer so it wont show up in model.parameters 
    BUT model.to() will move the buffers to the desired device
'''
class AdversarialForFree(nn.Module):
    def __init__(self, e, min=0, max=1,
                 auto_zero_grad: bool = True, e_s = None, random_init=False):

        super(AdversarialForFree, self).__init__()
        self.e_s = e if e_s is None else e_s
        self.e = e
        self.min, self.max = min, max
        self.auto_zero_grad = auto_zero_grad
        self.random_init = random_init

    def forward(self, x, auto_zero_grad=None):
        if auto_zero_grad is None:
            auto_zero_grad = self.auto_zero_grad

        if hasattr(self, 'm') and auto_zero_grad:
            self.zero_grad()

        if self.training:
            if not hasattr(self, 'm') or self.m.shape != x.shape:
                if self.random_init:
                    m = torch.rand(x.shape, device=x.device) * 2 * self.e - self.e
                    m.requires_grad_()
                else:
                    m = torch.zeros_like(x, device=x.device, requires_grad=True)
                self.register_buffer('m', m)
            return (x + self.m).clamp(self.min, self.max)
        else:
            return x

    def step(self):
        with torch.no_grad():
            self.m.grad.sign_()
            self.m += self.e_s * self.m.grad
            self.m.clamp_(-self.e, self.e)

    def zero_grad(self):
        if hasattr(self, 'm'):
            tensor_zero_grad(self.m)

    def clean(self):
        if hasattr(self, 'm'):
            delattr(self, 'm')
