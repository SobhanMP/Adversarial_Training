from calendar import EPOCH
from collections import defaultdict
import time
import torch
import torch.nn.functional as F


"""Set the gradient of a tensor to zero, just like torch.optim.Optimizer"""
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

""" A batched average tracker


"""
class Collectinator:
    def __init__(self, mean=0):
        self.n = 0
        self.mean = mean
    def add(self, v, m=1):
        nm = self.n + m
        self.mean = self.mean * (self.n / nm) + v * (m / nm)
        self.n = nm 


""" A performance tracker

Track accuracy, loss and runtime
PRINTING CAUSES THE END TIME TO BE SET (only the first print)
"""
class Logisticator:
    def __init__(self, epoch=None) -> None:
        self._acc = Collectinator()
        self._loss = Collectinator()
        self.acc = 0
        self.loss = 0
        self.now = time.time()
        self.end_time = None
        self.epoch = epoch
    
    def add(self, acc, loss, m):
        self._acc.add(acc, m)
        self.acc = self._acc.mean
        self._loss.add(loss, m)
        self.loss = self._loss.mean

    def __str__(self):
        self.end()
        return f'{self.loss:.4f} {self.acc * 100:0.1f}% {self.end_time - self.now:.1f}s'
    
    def end(self):
        if self.end_time is None:
            self.end_time = time.time()
            self.time = self.end_time - self.now

"""
Take the average but instead of reducing the dims, keep them

Is there a pytorch version already?
"""
def mean_keepdim(inputs, dims):
    d = [i for i in range(len(inputs.shape)) 
        if i not in dims]
    return inputs.mean(dim=d)

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, axis=1)
    return torch.sum(preds == labels).item() / len(preds)

def train_with_replay(K, model, trainloader, optimizer, epoch, 
    input_func=lambda x, y: x,
    after_func=lambda model: None):
    logs = Logisticator(epoch)
    
    model.train()

    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = map(lambda x: x.cuda(), data)
        for k in range(K):
            inputs = input_func(inputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()
            after_func(model)

            acc = accuracy(outputs, labels)
            logs.add(acc, loss.item(), inputs.size(0))
    print(f'train \t {epoch + 1}: {logs}')
    return logs

def run_val(model, testloader, epoch, name='val'):
    model.train(False)
    # valdiation loss
    with torch.no_grad():
        logs = Logisticator(epoch)
        
        for data in testloader:
            inputs, labels = map(lambda x: x.cuda(), data)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            acc = accuracy(outputs, labels)
            logs.add(acc, loss.item(), inputs.size(0))

        print(f'{name} \t {epoch + 1}: {logs}')
    return logs

'''
logholder is a default dict or similar object that gives a list for each attack
category is the name that should be appended to each attack
attacks is a list of attacks 
'''
def run_attacks(logholder, attacks, attack_names, model, testloader, epoch, category='adv_test'):
    model.train(False)
    for (attack, name) in zip(attacks, attack_names):
        logs = Logisticator(epoch)
        logholder[f'{category}/{name}'].append(logs)
        for data in testloader:
            inputs, labels = map(lambda x: x.cuda(), data)
            adv = attack(model, inputs, labels)

            with torch.no_grad():
                outputs = model(adv)
                loss = F.cross_entropy(outputs, labels)

                acc = accuracy(outputs, labels)
                logs.add(acc, loss.item(), inputs.size(0))
        print(f'{category}/{name} \t {epoch + 1}: {logs}')

def bimap(f, g, x):
    return [(f(a), g(b)) for (a, b) in x]
def identity(x):
    return x
def holder_to_dict(holder: defaultdict):
    return dict(bimap(identity, dict, holder.items()))