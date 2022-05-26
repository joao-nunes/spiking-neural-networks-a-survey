import numpy as np
import torch

class DropInputs(torch.nn.Module):

    def __init__(self, p_drop=0.1):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, img):
        shape = img.size()
        probs = [self.p_drop, (1-self.p_drop)]
        choices = np.array([1, 0])
        mask = np.ones(shape)
        mask = mask.flatten()
        for i in range(0, shape[1]**2):
            drop = np.random.choice(choices, p=probs)
            if drop:
                mask[i] = 0 * mask[i]
        return torch.tensor(mask*img.squeeze().numpy().flatten(),
                            dtype=torch.float).view(shape)

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
