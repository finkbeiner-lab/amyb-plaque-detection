import torch
import torch.optim


def step_with_grads(step_func):
  def step_wrap(self, closure=None, *args, grad_idxs, grad_fn, **kwargs):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    grad_groups = list()
    for idx, group in enumerate(self.param_groups):
      if idx in grad_idxs:
        grad_group = list()
        for p in group['params']:
          grad = None
          if p.grad is not None:
            grad = grad_fn(p.grad)
          grad_group.append(grad)
        grad_groups.append(grad_group)

    loss = step_func(self, closure=lambda: loss, *args, **kwargs)
    return loss, grad_groups
  return step_wrap




class GradSGD(torch.optim.SGD):
  @step_with_grads
  def step(self, *args, **kwargs):
    return super().step(*args, **kwargs)
