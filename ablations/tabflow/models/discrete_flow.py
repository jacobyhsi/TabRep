import torch

class BaseFlow():
    def __init__(self, num_classes):
        self.num_classes = num_classes

class DiscreteFlowMatcher(BaseFlow):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def sample_location_and_conditional_flow(self, x1, t=None):
        B, _ = x1.shape
        if t is None:
            t = torch.rand((B,))
        
        xt = x1.clone().long()
        for i, S in enumerate(self.num_classes):
            uniform_noise = torch.randint(0, S, (B,)).to(x1.device)
            corrupt_mask = torch.rand((B,)).to(x1.device) < (1 - t)
            xt[:, i][corrupt_mask] = uniform_noise[corrupt_mask]
        return xt
        