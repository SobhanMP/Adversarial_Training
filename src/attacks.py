from turtle import backward

import torch
import torch.nn.functional as F
from .utils import tensor_zero_grad, Logisticator
''' Projected gradient descent attack with k steps, M

model: an object that has .zero_grad() and __call__(batch) -> pred

criterion: a loss function that takes ONE argument
    use a lambda function to convert the l(x, y) to l(x)

x: an input batch
K: number of training steps
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

