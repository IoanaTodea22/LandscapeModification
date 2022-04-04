import torch
import torch.nn.functional as F
from torch.optim import Optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adagrad(params,
            grads,
            state_sums,
            state_steps,
            *,
            lr: float,
            weight_decay: float,
            lr_decay: float,
            eps: float):

    for (param, grad, state_sum, step) in zip(params, grads, state_sums, state_steps):
        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
            grad = grad.add(param, alpha=weight_decay)

        clr = lr / (1 + (step - 1) * lr_decay)
        state_sum.addcmul_(grad, grad, value=1)
        std = state_sum.sqrt().add_(eps)
        param.addcdiv_(grad, std, value=-clr)



class Adagrad_IKSA(Optimizer):

    def __init__(self, params, function, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, eps_iksa = 1):

        def f_def(x):
          return x

        defaults = dict(function = function, lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value, eps_iksa = eps_iksa)
        super(Adagrad_IKSA, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p, initial_accumulator_value, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @torch.no_grad()
    def step(self,c, running_loss, closure=None):
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
            state_sums = []
            state_steps = []
            func = group["function"]
            eps_iksa = group["eps_iksa"]

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grad = p.grad
                    new_grad = grad/(func(torch.max(torch.zeros(p.data.size(), device = device), running_loss - c)) + eps_iksa)
                    grads.append(new_grad)
                    state = self.state[p]
                    state_sums.append(state['sum'])
                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adagrad(params_with_grad,
                      grads,
                      state_sums,
                      state_steps,
                      lr=group['lr'],
                      weight_decay=group['weight_decay'],
                      lr_decay=group['lr_decay'],
                      eps=group['eps'])

        return loss