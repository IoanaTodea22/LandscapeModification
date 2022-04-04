import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rmsprop(params,
            grads,
            square_avgs,
            grad_avgs,
            momentum_buffer_list,
            *,
            lr: float,
            alpha: float,
            eps: float,
            weight_decay: float,
            momentum: float,
            centered: bool):

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
        else:
            avg = square_avg.sqrt().add_(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)
        else:
            param.addcdiv_(grad, avg, value=-lr)

class RMSprop_IKSA(Optimizer):

    def __init__(self, params, function, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, eps_iksa = 1):

        def f_def(x):
            return x

        defaults = dict(function = function, lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay, eps_iksa = 1)

        super(RMSprop_IKSA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop_IKSA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

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
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []
            func = group["function"]
            eps_iksa = group["eps_iksa"]

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')

                grad = p.grad
                new_grad = grad/(func(torch.max(torch.zeros(p.data.size(), device = device), running_loss - c)) + eps_iksa)
                grads.append(new_grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                state['step'] += 1


            rmsprop(params_with_grad,
                      grads,
                      square_avgs,
                      grad_avgs,
                      momentum_buffer_list,
                      lr=group['lr'],
                      alpha=group['alpha'],
                      eps=group['eps'],
                      weight_decay=group['weight_decay'],
                      momentum=group['momentum'],
                      centered=group['centered'])

        return loss