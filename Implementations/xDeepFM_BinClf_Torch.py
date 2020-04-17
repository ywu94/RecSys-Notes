import numpy as np
import torch
assert torch.__version__>='1.2.0', 'Expect PyTorch>=1.2.0 but get {}'.format(torch.__version__)
from torch import nn
import torch.nn.functional as F


class xDeepFM_Deep_Layer(nn.Module):
    """
    Deep component of xDeepFM
    """
    def __init__(self, input_dim, ffn_dim, ffn_dropout, *args, **kwargs):
        """
        : param input_dim: dimension of input data (dense feature + embedded sparse feature)
        : param ffn_dim: output dimensions for each feed forward network
        : param ffn_dropout: dropout ratios for each feed foreward network
        """
        super(xDeepFM_Deep_Layer, self).__init__(*args, **kwargs)
        assert isinstance(ffn_dim, list)
        assert isinstance(ffn_dropout, list) and 1+len(ffn_dim)==len(ffn_dropout)
        
        self.input_dim = input_dim
        self.ffn_dim = ffn_dim
        self.ffn_dropout = ffn_dropout
        
        setattr(self, 'batch_norm_input', nn.BatchNorm1d(input_dim))
        setattr(self, 'dropout_input', nn.Dropout(ffn_dropout[0]))
        
        for index, dims in enumerate(zip([input_dim]+ffn_dim[:-1], ffn_dim, ffn_dropout[1:]), start=1):
            inp_dim, out_dim, dropout_ratio = dims
            setattr(self, 'ffn_{}'.format(index), nn.Linear(inp_dim, out_dim))
            setattr(self, 'batch_norm_{}'.format(index), nn.BatchNorm1d(out_dim))
            setattr(self, 'dropout_{}'.format(index), nn.Dropout(dropout_ratio))
            
    def forward(self, inp):
        inp = getattr(self, 'batch_norm_input')(inp)                      # (None, input_dim)
        inp = F.relu(inp)                                                 # (None, input_dim)
        inp = getattr(self, 'dropout_input')(inp)                         # (None, input_dim)
        for i in range(1, len(self.ffn_dim)+1):                           # (None, input_dim)
            inp = getattr(self, 'ffn_{}'.format(i))(inp)                  # (None, ffn_dim[i-1])
            inp = getattr(self, 'batch_norm_{}'.format(i))(inp)           # (None, ffn_dim[i-1])
            inp = F.relu(inp)                                             # (None, ffn_dim[i-1])
            inp = getattr(self, 'dropout_{}'.format(i))(inp)              # (None, ffn_dim[i-1])
            
        output = inp                                                      # (None, ffn_dim[-1])

        return output

class xDeepFM_CIN_Layer(nn.Module):
    """
    Compressed Interaction Network for xDeepFM
    """
    def __init__(self, input_dim, feat_map_dim, split_half, activation=None, **kwargs):
        """
        : param input_dim: list/tuple of (n_feature, embedding_dim)
        : param feat_map_dim: dimension of feature maps for each layer in compressed interaction network
        : param split_half: whether to link half of feature maps to output
        : param activation: activation function to apply on feature map, original paper suggests linear activation works the best
        """
        super(xDeepFM_CIN_Layer, self).__init__(**kwargs)
        assert isinstance(input_dim, (list, tuple)) and len(input_dim)==2, 'Expect input_dim to be a list/tuple of length 2'
        assert isinstance(feat_map_dim, (list, tuple)) and len(feat_map_dim)>0, 'Expect feat_map_dim to be a list/tuple of length > 0'
        
        self.input_dim = input_dim
        self.n_feature, self.embedding_dim = input_dim
        self.feat_map_dim = feat_map_dim
        self.split_half = split_half
        self.activation = activation
        
        self.feat_map_link_dim = [self.n_feature]
        for i, dim in enumerate(feat_map_dim):
            setattr(self, 'conv1d_{}'.format(i), nn.Conv1d(self.feat_map_link_dim[-1]*self.feat_map_link_dim[0], dim, 1))
            if split_half and i != len(feat_map_dim)-1:
                assert dim%2==0, 'When split_half is set to True, all dims in feat_map_dim (except for last one) should be even'
                self.feat_map_link_dim.append(dim//2)
            else:
                self.feat_map_link_dim.append(dim)
        
    def forward(self, inp):
        inp_buf = [inp]
        out_buf = []
        
        for i, dim in enumerate(self.feat_map_dim):
            nxt = torch.einsum('ijk,izk->ijzk', inp_buf[-1], inp_buf[0])                                             # (None, H_i-1, H_0, embedding_dim)
            nxt = nxt.reshape(inp_buf[0].shape[0], inp_buf[-1].shape[1] * inp_buf[0].shape[1], self.embedding_dim)   # (None, H_i-1 * H_0, embedding_dim)
            nxt = getattr(self, 'conv1d_{}'.format(i))(nxt)                                                          # (None, H_i, embedding_dim)
            
            if self.activation is not None:
                nxt = self.activation(nxt)
                
            if self.split_half:
                if i != len(self.feat_map_dim)-1:
                    nxt, out = torch.split(nxt, [dim//2 for _ in range(2)], 1)
                else:
                    nxt, out = None, nxt
            else:
                nxt, out = nxt, nxt
                
            inp_buf.append(nxt)
            out_buf.append(out)
            
        out = torch.cat(out_buf, dim=1)                                                                              # (None, H_1+..+H_k, embedding_dim)
        out = torch.sum(out, dim=2)                                                                                  # (None, H_1+..+H_k)
        
        return out

class xDeepFM_Layer(nn.Module):
    """
    PyTorch implementation of extreme Deep FM for binary classification

    References
    [1]Paper: https://arxiv.org/pdf/1803.05170.pdf
    """
    def __init__(self, n_sparse_feature, sparse_embedding_dim, sparse_dim, dense_dim, ffn_dim, ffn_dropout, cin_dim, cin_split_half, reg_l2=1e-4, **kwargs):
        """
        : param n_sparse_feature: vocabulary size used for sparse feature embedding
        : param sparse_embedding_dim: dimension of embedding for sparse feature
        : param sparse_dim: dimension of sparse input
        : param dense_dim: dimension of dense input
        : param ffn_dim: output dimensions for each feed forward network
        : param ffn_dropout: dropout ratios for each feed foreward network
        : param cin_dim: dimension of feature maps for each layer in compressed interaction network
        : param cin_split_half: whether to link half of each hidden layer in compressed interaction network to output
        : param reg_l2: λ for l2 regularization, to be implemented
        """
        super(xDeepFM_Layer, self).__init__(**kwargs)
        assert isinstance(ffn_dim, list) and len(ffn_dim) > 0, 'Invalid setup for ffn layer'
        assert isinstance(cin_dim, list) and len(cin_dim) > 0, 'Invalid setup for cin layer'
        
        self.n_sparse_feature = n_sparse_feature
        self.sparse_embedding_dim = sparse_embedding_dim
        self.sparse_dim = sparse_dim
        self.dense_dim = dense_dim
        self.ffn_dim = ffn_dim
        self.ffn_dropout = ffn_dropout
        self.cin_dim = cin_dim
        self.cin_split_half = cin_split_half
        self.reg_l2 = reg_l2
        
        # Sparse Feature Embedding
        self.sparse_embedding = nn.Embedding(n_sparse_feature, sparse_embedding_dim)
        nn.init.xavier_normal_(self.sparse_embedding.weight)
        
        # Linear Component
        self.lin_dense_weights = nn.Parameter(torch.zeros(dense_dim, 1))
        nn.init.xavier_normal_(self.lin_dense_weights)
        self.lin_sparse_weights = nn.Embedding(n_sparse_feature, 1)
        nn.init.xavier_normal_(self.lin_sparse_weights.weight)
        
        # Deep Component
        self.deep_inp_dim = sparse_embedding_dim * sparse_dim + dense_dim
        self.deep_layer = xDeepFM_Deep_Layer(self.deep_inp_dim, ffn_dim, ffn_dropout)
        self.deep_out_dim = ffn_dim[-1]
        
        # CIN Component
        self.cin_inp_dim = (sparse_dim, sparse_embedding_dim)
        self.cin_layer = xDeepFM_CIN_Layer(self.cin_inp_dim, cin_dim, cin_split_half)
        self.cin_out_dim = sum([i//2 for i in cin_dim[:-1]]) + cin_dim[-1] if cin_split_half else sum(cin_dim)
        
        # Output 
        self.final_inp_dim = 1 + self.deep_out_dim + self.cin_out_dim
        self.final_ffn = nn.Linear(self.final_inp_dim, 1)
        
    def forward(self, inp_dense, inp_sparse):
        # Embed Sparse Feature
        inp_sparse_embed = self.sparse_embedding(inp_sparse)                             # (None, sparse_dim) -> (None, sparse_dim, sparse_embedding_dim)
        inp_sparse_embed_flatten = torch.flatten(inp_sparse_embed, start_dim=1)          # (None, sparse_dim * sparse_embedding_dim)
        
        # Linear Output
        lin_dense_inp = torch.unsqueeze(inp_dense, 1)                                    # (None, dense_dim) -> (None, 1, dense_dim)
        lin_dense_out = torch.matmul(lin_dense_inp, self.lin_dense_weights)              # (None, 1, dense_dim), (dense_dim, 1) -> (None, 1, 1)
        lin_dense_out = torch.squeeze(lin_dense_out, 2)                                  # (None, 1, 1) -> (None, 1)
        lin_sparse_out = self.lin_sparse_weights(inp_sparse)                             # (None, sparse_dim) -> (None, sparse_dim, 1)
        lin_sparse_out = torch.sum(lin_sparse_out, dim=1)                                # (None, sparse_dim, 1) -> (None, 1)
        lin_out = torch.add(lin_dense_out, lin_sparse_out)                               # (None, 1)
        
        # Deep Output
        deep_inp = torch.cat((inp_dense, inp_sparse_embed_flatten), dim=1)               # (None, dense_dim + sparse_dim * sparse_embedding_dim)
        deep_out = self.deep_layer(deep_inp)                                             # (None, self.deep_out_dim)
        
        # CIN Output
        cin_inp = inp_sparse_embed                                                       # (None, sparse_dim, sparse_embedding_dim)
        cin_out = self.cin_layer(cin_inp)                                                # (None, self.cin_out_dim)
        
        # Output
        output = torch.cat((lin_out, deep_out, cin_out), dim=1)                          # (None, self.final_inp_dim)
        output = self.final_ffn(output)                                                  # (None, 1)
        
        return output
            
        
        