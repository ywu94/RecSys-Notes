"""
PyTorch implementation of Factorization Machine of degree d = 2 for binary classification

References
[1]Paper: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
[2]JianzhouZhan Git: https://github.com/JianzhouZhan/Awesome-RecSystem-Models/blob/master/Model/FM_PyTorch.py 
"""

import torch
assert torch.__version__>='1.2.0', 'Expect PyTorch>=1.2.0 but get {}'.format(torch.__version__)
from torch import nn

class FM_2D_Layer(nn.Module):
    def __init__(self, n_feature, n_field, embedding_dim, *args, **kwargs):
        super(FM_2D_Layer, self).__init__(*args, **kwargs)
        self.n_feature = n_feature    
        self.n_field = n_field
        self.embedding_dim = embedding_dim
        
        # Weights for first degree features (None, n_feature, 1)
        self.feature_weight = nn.Embedding(n_feature, 1)                    
        nn.init.xavier_normal_(self.feature_weight.weight)
        
        # Weights for second degree interaction features (None, n_feature, embedding_dim)
        self.interaction_weight = nn.Embedding(n_feature, embedding_dim)
        nn.init.xavier_normal_(self.interaction_weight.weight)
        
        # Bias (None, 1)
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, feature_index, feature_value):
        # Contribution from first degree features
        feature_value = torch.unsqueeze(feature_value, 2)                       # (None, n_field) -> (None, n_field, 1)
        feature_weight = self.feature_weight(feature_index)                     # (None, n_field) -> (None, n_field, 1)
        
        first_res = torch.mul(feature_value, feature_weight)                    # (None, n_field, 1)
        first_res = torch.squeeze(first_res, 2)                                 # (None, n_field, 1) -> (None, n_field)
        first_res = torch.sum(first_res, 1)                                     # (None, n_field) -> (None,)
        
        # Contribution from second degree interaction features
        interaction_weight = self.interaction_weight(feature_index)             # (None, n_field) -> (None, n_field, embedding_dim)
        feature_value_embed = torch.mul(interaction_weight, feature_value)      # (None, n_field, embedding_dim)
        
        int_sum_square = torch.sum(feature_value_embed, 1)                      # (None, embedding_dim)
        int_sum_square = torch.pow(int_sum_square, 2)                           # (None, embedding_dim)
        
        int_square_sum = torch.pow(feature_value_embed, 2)                      # (None, n_field, embedding_dim)
        int_square_sum = torch.sum(int_square_sum, 1)                           # (None, embedding_dim)
        
        second_res = 0.5*torch.sub(int_sum_square, int_square_sum)              # (None, embedding_dim)
        second_res = torch.sum(second_res, 1)                                   # (None, embedding_dim) -> (None,)
        
        # Final Output
        output = torch.unsqueeze(first_res + second_res + self.bias, 1)         # (None,) -> (None, 1)
        return output