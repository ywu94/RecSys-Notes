"""
PyTorch implementation of Deep Cross Network for binary classification

References
[1]Paper: https://arxiv.org/pdf/1708.05123.pdf
"""

import torch
assert torch.__version__>='1.2.0', 'Expect PyTorch>=1.2.0 but get {}'.format(torch.__version__)
from torch import nn
import torch.nn.functional as F

class DCN_Layer(nn.Module):
    def __init__(self):
        