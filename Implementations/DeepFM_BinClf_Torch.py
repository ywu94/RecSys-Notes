import torch
assert torch.__version__>='1.2.0', 'Expect PyTorch>=1.2.0 but get {}'.format(torch.__version__)
from torch import nn
import torch.nn.functional as F

class DeepFM_2D_Layer(nn.Module):
    """
    PyTorch implementation of DeepFM of degree d = 2 for binary classification

    References
    [1]Paper: https://www.ijcai.org/Proceedings/2017/0239.pdf
    """
    def __init__(self, n_feature, n_field, embedding_dim, ffn_size, fm_dropout, ffn_dropout, reg_l1=0, reg_l2=0, **kwargs):
        """
        : param n_feature: vocabulary size used for feature embedding (both dense and sparse)
        : param n_field: number of fields each input will have
        : param embedding_dim: dimension of embedding
        : param ffn_size: output dimensions for each feed forward network
        : param fm_dropout: dropout ratio for first degree and second degree output of FM
        : param ffn_dropout: dropout ratio for each feed forward network
        : param reg_l1: parameter for l1 regularization
        : param reg_l2: parameter for l1 regularization
        """
        super(DeepFM_2D_Layer, self).__init__(**kwargs)
        assert isinstance(ffn_size, list)
        assert isinstance(fm_dropout, list) and len(fm_dropout)==2
        assert isinstance(ffn_dropout, list) and 1+len(ffn_size)==len(ffn_dropout)
        
        self.n_feature = n_feature    
        self.n_field = n_field
        self.embedding_dim = embedding_dim
        self.ffn_size = ffn_size
        self.fm_dropout = fm_dropout
        self.ffn_dropout = ffn_dropout
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        
        # Weights for first degree features (None, n_feature, 1)
        self.first_degree_weight = nn.Embedding(n_feature, 1)                    
        nn.init.xavier_normal_(self.first_degree_weight.weight)
        
        # Weights for feature imbedding (None, n_feature, embedding_dim)
        self.feature_embedding_weight = nn.Embedding(n_feature, embedding_dim)
        nn.init.xavier_normal_(self.feature_embedding_weight.weight)
        
        # Deep Network
        nn_dim = [n_field * embedding_dim] + ffn_size
        for i in range(len(ffn_size)):
            setattr(self, 'ffn_{}'.format(i), nn.Linear(nn_dim[i], nn_dim[i+1]))
            setattr(self, 'batch_norm_{}'.format(i), nn.BatchNorm1d(nn_dim[i+1]))
            setattr(self, 'dropout_{}'.format(i), nn.Dropout(ffn_dropout[i+1]))
            
        # Combine
        self.final_ffn = nn.Linear(n_field + embedding_dim + ffn_size[-1], 1)
        
        # All Other Dropout Layers
        self.first_res_dropout = nn.Dropout(self.fm_dropout[0])
        self.second_res_dropout = nn.Dropout(self.fm_dropout[1])
        self.dense_dropout = nn.Dropout(self.ffn_dropout[0])
        
    def forward(self, feature_index, feature_value):
        # Process input
        feature_value = torch.unsqueeze(feature_value, 2)                         # (None, n_field) -> (None, n_field, 1)
        first_degree_weight = self.first_degree_weight(feature_index)             # (None, n_field) -> (None, n_field, 1)
        feature_embedding_weight = self.feature_embedding_weight(feature_index)   # (None, n_field) -> (None, n_field, embedding_dim)
        feature_value_embed = torch.mul(feature_embedding_weight, feature_value)  # (None, n_field, embedding_dim)
        
        # Calculate first degree output of FM
        first_res = torch.mul(feature_value, first_degree_weight)                 # (None, n_field, 1)
        first_res = torch.squeeze(first_res, 2)                                   # (None, n_field, 1) -> (None, n_field)
        first_res = self.first_res_dropout(first_res)                             # (None, n_field)
        
        # Calculate second degree output of FM
        int_sum_square = torch.sum(feature_value_embed, 1)                        # (None, embedding_dim)
        int_sum_square = torch.pow(int_sum_square, 2)                             # (None, embedding_dim)
        int_square_sum = torch.pow(feature_value_embed, 2)                        # (None, n_field, embedding_dim)
        int_square_sum = torch.sum(int_square_sum, 1)                             # (None, embedding_dim)
        second_res = 0.5*torch.sub(int_sum_square, int_square_sum)                # (None, embedding_dim)
        second_res = self.second_res_dropout(second_res)                          # (None, embedding_dim)
        
        # Calculate deep output
        inp_deep = torch.flatten(feature_value_embed, start_dim=1)                # (None, n_field * embedding_dim)
        inp_deep = self.dense_dropout(inp_deep)                                   # (None, n_field * embedding_dim)
        for i in range(len(self.ffn_size)):
            inp_deep = getattr(self, 'ffn_{}'.format(i))(inp_deep)
            inp_deep = getattr(self, 'batch_norm_{}'.format(i))(inp_deep)
            inp_deep = F.relu(inp_deep)
            inp_deep = getattr(self, 'dropout_{}'.format(i))(inp_deep)
         
        # Calculate final output
        inp = torch.cat((first_res, second_res, inp_deep), dim = 1)               # (None, n_field + embedding_dim + ffn_size[-1])
        output = self.final_ffn(inp)                                              # (None, n_field + embedding_dim + ffn_size[-1]) -> (None, 1)
        
        return output