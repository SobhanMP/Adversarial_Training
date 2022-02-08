from turtle import backward

import torch
import torch.nn.functional as F
from .utils import tensor_zero_grad, Logisticator
''' 
Projected gradient descent attacks with K steps on the ininity ball 
model: an object that has .zero_grad() and __call__(batch) -> pred
K: number of training steps
e and e_s are the maximim noise level
min-max, set the levels that the output should be clamped into
'''

class Attack:
    def __init__(self, K, e, e_s, min=0, max=1) -> None:
        self.K = K
        self.e = e
        self.e_s = e_s
        self.min = min
        self.max = max
    
    def step(self, noise):
        with torch.no_grad():
                noise.grad.sign_()
                noise += self.e_s * noise.grad
                noise.clamp_(-self.e, self.e)

    def zero(self, model, noise):
        model.zero_grad()
        tensor_zero_grad(noise)
    
    def init_noise(self, x):
        a = torch.rand(x.shape, device=x.device) * 2 * self.e - self.e
        a.requires_grad_()
        return a

    def run(self, model, x, noise):
        return model(self.adv(x, noise))
    
    def adv(self, x, noise):
        return (x + noise).clamp(self.min, self.max)

    def finalize(self, model, noise):
        noise.requires_grad_(False)
        model.train(self.t)
        
    
    def initialize(self, model):
        self.t = model.training # for 7-PDG training
        model.train(False)

'''
PGD attack
e is the noise level and e_s is the step size.
early_stopping will cause the algorithm to train early if all predictions are wrong, useful for testing accuracy not loss
'''
class PGD(Attack):
    def __init__(self, K, e, e_s, min=0, max=1, loss=F.cross_entropy, early_stopping=False) -> None:
        super().__init__(K, e, e_s, min, max)
        self.early_stopping = early_stopping
        self.loss = loss

    def __call__(self, model, x, y):
        noise = self.init_noise(x)
        self.initialize(model)
        
        for _ in range(self.K):
            self.zero(model, noise)    
            
            p = self.run(model, x, noise)
            
            if self.early_stopping:
                pm = p.max(axis=1)[0]
                py = p[torch.arange(x.size(0)), y] 
                if (py < pm).all():
                    break
            
            loss = self.loss(p, y)
            loss.backward()
            
            self.step(noise)
            
        self.zero(model, noise)
        self.finalize(model, noise)
        
        return self.adv(x, noise)
    
'''
PGD attack with Carlini-Wagner loss. 
e is the noise level and e_s is the step size
based on the version from the free repo

'''
class CW(Attack):
    def __init__(self, K, c, e, e_s, min=0, max=1) -> None:
        super().__init__(K, e, e_s, min, max)
        self.c = c
    
    def __call__(self, model, x, y):
        noise = self.init_noise(x)
        self.initialize(model)
        mask = None
        
        for _ in range(self.K):
            self.zero(model, noise)

            p = self.run(model, x, noise)
            
            if mask is None:
                mask = torch.eye(p.shape[1], device=x.device)[y, :]

            correct_logit = p[torch.arange(p.shape[0]), y]
            wrong_logit = ((1 - mask) * p - self.c * mask).max(axis=1)[0]
            loss = -F.relu(correct_logit - wrong_logit).mean()
            
            loss.backward()

            self.step(noise)

        self.zero(model, noise)
        self.finalize(model, noise)
        return self.adv(x, noise)

