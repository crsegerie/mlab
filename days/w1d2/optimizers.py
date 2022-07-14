"""In this script, we implement 3 optimizers

https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c
"""

import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple
import optimizers_tests as tests


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


def _train(model: nn.Module, dataloader: DataLoader, lr, momentum):
    opt = torch.optim.SGD(model.parameters(), lr, momentum)
    for X, y in dataloader:
        opt.zero_grad()
        pred = model(X)
        loss = F.l1_loss(pred, y)
        loss.backward()
        opt.step()
    return model


def _accuracy(model: nn.Module, dataloader: DataLoader):
    n_correct = 0
    n_total = 0
    for X, y in dataloader:
        n_correct += (model(X).argmax(1) == y).sum().item()
        n_total += len(y)
    return n_correct / n_total


def _evaluate(model: nn.Module, dataloader: DataLoader):
    sum_abs = 0.0
    n_elems = 0
    for X, y in dataloader:
        sum_abs += (model(X) - y).abs().sum()
        n_elems += y.shape[0] * y.shape[1]
    return sum_abs / n_elems


def _rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


def _opt_rosenbrock(xy, lr, momentum, n_iter):
    w_history = torch.zeros([n_iter + 1, 2])
    w_history[0] = xy.detach()
    opt = torch.optim.SGD([xy], lr=lr, momentum=momentum)

    for i in range(n_iter):
        opt.zero_grad()
        _rosenbrock(xy[0], xy[1]).backward()

        opt.step()
        w_history[i + 1] = xy.detach()
    return w_history


"""
1. Read the _opt_rosenbrock code. What are the methods in the SGD class?
2. Why do we have to zero the gradient in pytorch?
2. Implement zero_grad. In pytorch, to zero a gradient means assigning None.
3. Implement step.
    3.1 What is the formula of the update when there is some weight_decay? Assume wd absorbs the constant.
    3.2 Separate the cases self.momentum equals zero or not.
    3.3 Don't forget the 'with torch.no_grad()' context manager.
"""


class _SGD:
    def __init__(
        self, params, lr: float, momentum: float, dampening: float, weight_decay: float
    ):
        self.params = list(params)
        self.lr = lr
        self.wd = weight_decay
        self.momentum = momentum  # here, it's the correct definition of momentum
        self.dampening = dampening  # generally 1-momentum = 1-dampening
        self.b = [None for _ in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step_(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                ...


"""
_RMSprop: Using the square of the gradient to adapt the lr
- What is the formula of the update when there is some weight_decay? Assume wd absorbs the constant.
- Update the squared gradient.
- Why do we use the gradient squared? Why do we say that the lr in _RMSprop is adaptive?
- eps should be outside the squared root. How would you adapt eps if it were inside?
- Separate the cases self.momentum zero or not.
"""


class _RMSprop:
    def __init__(
        self,
        params,
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha  # momentum of gradient squared
        self.eps = eps
        self.wd = weight_decay
        self.momentum = momentum

        self.b2 = [torch.zeros_like(p) for p in self.params]  # gradient squared
        self.b = [torch.zeros_like(p) for p in self.params]  # gradient

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                ...


"""
Adam, by far the most used optimizer.
It's a combination of SGD + RMSProps and uses one momentum for the gradient, and another for the gradient squared.
- update b1, b2 
- compute b1_hat, b2_hat    
    - b1_hat = self.b1[i] / (1.0 - self.beta1**self.t)
    - same for b2_hat
    - Why this formula?
- update the gradient
"""


class _Adam:
    def __init__(
        self,
        params,
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas  # momenti of b1 and b2
        self.eps = eps
        self.wd = weight_decay

        self.b1 = [torch.zeros_like(p) for p in self.params]
        self.b2 = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                ...


"""
Bonus: 
- Give a reason to use SGD instead of Adam.
- What is an abstract class in python?
- Modify the script to use an abstract class.
- What is a Parent class? Modify the script to use a Parent class
- What is a context manager?
"""

if __name__ == "__main__":
    tests.test_mlp(_MLP)
    tests.test_train(_train)
    tests.test_accuracy(_accuracy)
    tests.test_evaluate(_evaluate)
    tests.test_rosenbrock(_opt_rosenbrock)
    tests.test_sgd(_SGD)
    tests.test_rmsprop(_RMSprop)
    tests.test_adam(_Adam)
