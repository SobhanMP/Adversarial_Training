# Implementation of [Adversarial Training for Free!](https://arxiv.org/abs/1904.12843)

This is a reimplmentation from scratch of the paper "Adversarial Training for Free!" in the context of the cours [IFT 6756 - Game Theory and ML course](https://gauthiergidel.github.io/courses/game_theory_ML_2021.html).

The main implementation is in the src folder the other notebooks test the paper a bit.

The notebooks contain a few tests. 

The `cifar` notebook is a test on CIFAR-10 with a wide resnet where we test paper. It compares training with replay, adversarial training and training for free. 


An example on the fashion mnist dataset is in the `small_batch` branch. Due to some parallel work, it 
uses an older version of the code hence why it's in a branch. The goal was to train a model for "free" with small batches since the author use large batches. 


Last but not learst, the `audio` notebook is a an example of audio classification with a network that resembles the M5 network with two of the conolution layers replaced with resnet blocks. I used this last layer to do some more in depth testing because testing with CIFAR is a bit time consuming.


The pytorch layer `AdversarialForFree` can be added to any model with `nn.Sequential` to get adversarial trainig for free. Honestly this just felt like a wasted opportunity by the authors. Just remember to call `.step()` after each gradient calculation, i've wasted too many hours wondering why no training is getting done. 


## Pain points

I ran into a few problem coding this up. For instance i ended up readin the neurips version which does not have the psudocode. Wide Reset 32 layers, 10 times wider does not seem to exist, according the default implementation it should have $(d - 4) \% = 0$ layers. The Carlini-Wagner loss is not really the Carlini-Wagner loss as their implementation was a bit more complicated (for better or worse they don't use PGD to solve the problem). here we used to the same algorithm as the one in the main repo. Clamping the adversarial example in the valid domain is not mentioned in the psudocode but is present in the implementation so we kept it. We weren't sure what PGD with restarts were so we ignored them. The learning rate for PGD $2.5\epsilon/K$ was a bit triccky to find.


## Remarks 

This layer makes the training a bit harder, therefore it's not for free though. In fact i claim that it makes training harder than normal adversarial training. Supprisingly, even though training with replay reduces the cpu-gpu data transfer, it doesn't make it faster. I guess this is due to pytorch's (and cuda's?) async nature. The biggest downside of this method is the lack of control over how good the model should be against adversarial examples and the cost of training with replays.

The results on the cifar 10 dataset can be seen below. FSGM (large step) uses a $2\epsilon$ step size while the counter part uses a $0.9\epsilon$. Most of this table can be generated with the `cifar` notebook. All of the times are for a gtx-3090.

![cifar table](figures/cifar.png)

As it can been seen, it's not clear in this example whether training for free is worth using or not. The most interesting case are those two example that take less than hour to train. It's possible to reach interesting levels of performance with both training methods in a short amount of time. But even in the best case for the training for free model (m=10), the performance and time is pretty close to the the time needed to train the PGD-2 model. To me it's unclear which is better specially for very short training time




