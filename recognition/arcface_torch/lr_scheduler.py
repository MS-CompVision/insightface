import math
import torch

class PolynomialLRWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Polynomial LR with linear warmup.

    Expected keyword names (to match your train script):
      - warmup_iters (int)
      - total_iters (int)

    Usage (example from your train_v2.py):
      lr_scheduler = PolynomialLRWarmup(
          optimizer=opt,
          warmup_iters=cfg.warmup_step,
          total_iters=cfg.total_step
      )
    """

    def __init__(self,
                 optimizer,
                 warmup_iters: int = 0,
                 total_iters: int = 1000,
                 power: float = 1.0,
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.warmup_iters = int(warmup_iters)
        self.total_iters = int(total_iters)
        self.power = float(power)

        # safe super init for PyTorch versions with/without verbose arg
        try:
            super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        except TypeError:
            super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # self.last_epoch is maintained by base class
        current_iter = max(0, self.last_epoch)

        if current_iter < self.warmup_iters and self.warmup_iters > 0:
            # linear warmup from 0 -> base_lr
            warmup_factor = (current_iter + 1) / float(self.warmup_iters)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # after warmup, polynomial decay to 0 at total_iters
        if current_iter >= self.total_iters:
            return [0.0 for _ in self.base_lrs]

        # decay progress: map current_iter from [warmup_iters, total_iters) -> [0,1)
        if self.total_iters - self.warmup_iters > 0:
            progress = (current_iter - self.warmup_iters) / float(self.total_iters - self.warmup_iters)
            factor = (1.0 - progress) ** self.power
        else:
            factor = 1.0

        return [base_lr * factor for base_lr in self.base_lrs]

    # wrapper compat: older/newer torch may expect get_last_lr()
    def get_last_lr(self):
        return self.get_lr()

