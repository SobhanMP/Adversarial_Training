import torch
from .utils import tensor_zero_grad
'''
Projected gradient descent attack with k steps
'''
def PGK(model, criterion, x, e, K, min=0, max=1):
    noise = torch.zeros_like(x, device=x.device, requires_grad=True)
    for k in range(K):
        model.zero_grad()
        tensor_zero_grad(noise)

        outputs = model((x + noise).clamp(min, max))

        loss = criterion(outputs)
        loss.backward()

        with torch.no_grad():
            noise.grad.sign_()
            noise += e * noise.grad
            noise.clamp_(-e, e)
    
    return noise
    