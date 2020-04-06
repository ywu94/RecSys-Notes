import torch
assert torch.__version__>='1.2.0', 'Expect PyTorch>=1.2.0 but get {}'.format(torch.__version__)
from torch import nn
import torch.nn.functional as F

class DCN_Deep_Layer(nn.Module):
    """
    Deep component of DCN
    """
    def __init__(self, input_dim, ffn_dim, ffn_dropout, *args, **kwargs):
        """
        : param input_dim: dimension of input data (dense feature + embedded sparse feature)
        : param ffn_dim: output dimensions for each feed forward network
        : param ffn_dropout: dropout ratios for each feed foreward network
        """
        super(DCN_Deep_Layer, self).__init__(*args, **kwargs)
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
        
class DCN_Cross_Layer(nn.Module):
    """
    Cross component of DCN
    """
    def __init__(self, input_dim, n_cross_op, *args, **kwargs):
        """
        : param input_dim: dimension of input data (dense feature + embedded sparse feature)
        : param n_cross_op: number of cross op to perform
        """
        super(DCN_Cross_Layer, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.n_cross_op = n_cross_op
        
        for i in range(1, n_cross_op+1):
            setattr(self, 'weight_{}'.format(i), nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            setattr(self, 'bias_{}'.format(i), nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            setattr(self, 'batch_norm_{}'.format(i), nn.BatchNorm1d(input_dim, affine=True))
        
    def forward(self, inp):
        output = inp                                                                         # (None, input_dim)
        inp = torch.unsqueeze(inp, 2)                                                        # (None, input_dim) -> (None, input_dim, 1)
        
        for i in range(1, self.n_cross_op+1):
            nxt_output = torch.unsqueeze(output, 1)                                          # (None, input_dim) -> (None, 1, input_dim)
            nxt_output = torch.bmm(inp, nxt_output)                                          # (None, input_dim, 1), (None, 1, input_dim) -> (None, input_dim, input_dim)
            nxt_output = torch.matmul(nxt_output, getattr(self, 'weight_{}'.format(i)))      # (None, input_dim, input_dim), (input_dim) -> (None, input_dim)
            nxt_output = nxt_output + getattr(self, 'bias_{}'.format(i))                     # (None, input_dim)
            output = getattr(self, 'batch_norm_{}'.format(i))(output + nxt_output)           # (None, input_dim)
            
        return output
    
    
class DCN_Layer(nn.Module):
    """
    PyTorch implementation of Deep Cross Network for binary classification

    References
    [1]Paper: https://arxiv.org/pdf/1708.05123.pdf
    """
    def __init__(self, n_sparse_feature, sparse_embedding_dim, sparse_dim, dense_dim, ffn_dim, ffn_dropout, n_cross_op, *args, **kwargs):
        """
        : param n_sparse_feature: vocabulary size used for sparse feature embedding
        : param sparse_embedding_dim: dimension of embedding for sparse feature
        : param sparse_dim: dimension of sparse input
        : param dense_dim: dimension of dense input
        : param ffn_dim: output dimensions for each feed forward network
        : param ffn_dropout: dropout ratios for each feed foreward network
        : param n_cross_op: number of cross op to perform
        """
        super(DCN_Layer, self).__init__(*args, **kwargs)
        assert isinstance(ffn_dim, list) and len(ffn_dim) > 0, 'Invalid setup for deep layer'
        assert isinstance(n_cross_op, int) and n_cross_op > 0, 'Invalid setup for cross layer'
        
        self.n_sparse_feature = n_sparse_feature
        self.sparse_embedding_dim = sparse_embedding_dim
        self.sparse_dim = sparse_dim
        self.dense_dim = dense_dim
        self.ffn_dim = ffn_dim
        self.ffn_dropout = ffn_dropout
        self.n_cross_op = n_cross_op
        
        self.sparse_embedding = nn.Embedding(n_sparse_feature, sparse_embedding_dim)
        nn.init.xavier_normal_(self.sparse_embedding.weight)
        
        self.input_dim = sparse_dim * sparse_embedding_dim + dense_dim
        
        self.input_bn = nn.BatchNorm1d(self.input_dim, affine=False)
        self.deep_layer = DCN_Deep_Layer(self.input_dim, ffn_dim, ffn_dropout)
        self.cross_layer = DCN_Cross_Layer(self.input_dim, n_cross_op)
        
        self.final_ffn = nn.Linear(self.input_dim + ffn_dim[-1], 1)
        
    def forward(self, inp_dense, inp_sparse):
        inp_sparse_embed = torch.flatten(self.sparse_embedding(inp_sparse), start_dim=1) # (None, sparse_dim) -> (None, sparse_dim * sparse_embedding_dim)
        inp = torch.cat((inp_dense, inp_sparse_embed), 1)                                # (None, dense_dim + sparse_dim * sparse_embedding_dim)
        inp = self.input_bn(inp)                                                         # (None, dense_dim + sparse_dim * sparse_embedding_dim)
        
        deep_out = self.deep_layer(inp)                                                  # (None, ffn_dim[-1])
        cross_out = self.cross_layer(inp)                                                # (None, dense_dim + sparse_dim * sparse_embedding_dim)
        output = torch.cat((deep_out, cross_out), 1)                                     # (None, dense_dim + sparse_dim * sparse_embedding_dim + ffn_dim[-1])
    
        output = self.final_ffn(output)                                                  # (None, 1)
        
        return output
            

        