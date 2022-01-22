import time
import torch

def tensor_zero_grad(x, set_to_none: bool = False):
    if x.grad is not None:
        if set_to_none:
            x.grad = None
        else:
            if x.grad.grad_fn is not None:
                x.grad.detach_()
            else:
                x.grad.requires_grad_(False)
            x.grad.zero_()
    """
copyright doofenshmirtz co.
"""
class Collectinator:
    def __init__(self, mean=0):
        self.n = 0
        self.mean = mean
    def add(self, v, m=1):
        nm = self.n + m
        self.mean = self.mean * (self.n / nm) + v * (m / nm)
        self.n = nm 

class Logisticator:
    def __init__(self) -> None:
        self.acc = Collectinator()
        self.loss = Collectinator()
        self.now = time.time()
    
    def add(self, acc, loss, m):
        self.acc.add(acc, m)
        self.loss.add(loss, m)

    def __str__(self):
        return f'{self.loss.mean:.4f} {self.acc.mean * 100:0.1f}% {time.time() - self.now:.1f}s'

def mean_keepdim(inputs, dims):
    d = [i for i in range(len(inputs.shape)) 
        if i not in dims]
    return inputs.mean(dim=d)

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, axis=1)
    return torch.sum(preds == labels).item() / len(preds)