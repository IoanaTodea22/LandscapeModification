"""
In this module we use the Pytorch version of the SGD implementation and we modify it add IAKSA.
Source: https://github.com/pytorch/pytorch

Essential modifications are marked with comments.
IAKSA parameters to consider:
Class paramaters:
- f(x) - defualt f(x) = x
- eps_IAKSA - defualt eps_IAKSA = 1

Step parameters:
- c
- running_loss - U(x) in theory
"""

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import random
#torch.manual_seed(0)


# if available, we want to operate on a gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sgd(params,
        d_p_list,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):


    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


class SGD_IKSA(Optimizer):

    def __init__(self, params, function, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps_IAKSA=1):

        # by defualt we consider
        def f_def(x):
          return x

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, function = function, eps_IAKSA=eps_IAKSA)

        super(SGD_IKSA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_IKSA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, c, running_loss, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            func = group["function"]
            eps_IAKSA = group["eps_IAKSA"]

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grad = p.grad
                    new_grad = grad/(func(torch.max(torch.zeros(p.data.size(), device = device), running_loss - c)) + eps_IAKSA)
                    #print("running_loss-c", running_loss - c)
                    #print(func(torch.max(torch.zeros(p.data.size(), device = device), running_loss - c)))
                    d_p_list.append(new_grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
