# Implementation of [Adversarial Training for Free!](https://arxiv.org/abs/1904.12843)

the implementation is in the src folder and the pytorch layer `AdversarialForFree` does the work, see the documentation in the code for more information.

The notebooks contain a few tests. The `con` notebook is a test on CIFAR-10 with a wide resnet where we test the properties mentioned with `m=4` and compare it with projected gradient descent adversarial training on performance and time. Furtheremore, we also investigate the cost of training with replays.