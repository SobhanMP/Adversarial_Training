import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

"""
shift the average to 0 and the variance to 1
"""
class StandardScalerLayer(nn.Module):
    """
    data: function that returns an iterator
    keep_axis: iterable for axixes to skeep defaults keep second argument
    """
    def __init__(self, data, keep_dims=[1]):
        super(StandardScalerLayer, self).__init__()
        
        self.keep_dims=keep_dims

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

'''
add adversarial noise to the input,
you need to call .step AFTER updating the gradient
m needs to be a buffer so that it doesn't get updated,
changing input shapes causes the buffer to reset.
set min and max to meaning ful values or (infty)
'''
class AdversarialForFree(nn.Module):
    def __init__(self, e, min=0, max=1):
        super(AdversarialForFree, self).__init__()
        self.e = e
        self.min, self.max = min, max

    def forward(self, x, auto_zero_grad=True):
        if hasattr(self, 'm') and auto_zero_grad:
            self.zero_grad()

        if self.training:
            if not hasattr(self, 'm') or self.m.shape != x.shape:
                m = torch.zeros_like(x, device=x.device, requires_grad=True)
                self.register_buffer('m', m)
            return (x + self.m).clamp(self.min, self.max)
        else:
            return x
    
    def step(self):
        with torch.no_grad():
            self.m.grad.sign_()
            self.m += self.e * self.m.grad
            self.m.clamp_(-self.e, self.e)

    def zero_grad(self):
        if hasattr(self, 'm'):
            tensor_zero_grad(self.m)
    
    def clean(self):
        if hasattr(self, 'm'):
            delattr(self, 'm')