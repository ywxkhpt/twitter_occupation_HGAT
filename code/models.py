import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter
from functools import reduce
from utils import dense_tensor_to_sparse


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.normalize = False
#         self.attention = False
#
#         self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
#         self.gc2 = GraphConvolution(nhid, nclass, bias=False)
#
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = self.gc1(x, adj)
#         if self.attention:
#             weights = Attention_SupLevel()
#         if self.normalize:
#             x = F.normalize(x, p=2, dim=1)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         if self.normalize:
#             x = F.normalize(x, p=2, dim=1)
#         x = F.relu(x)
#         return F.log_softmax(x, dim=1)


class HGAT(nn.Module):
    def __init__(self, nfeat_list, nhid, nclass, dropout):
        super(HGAT, self).__init__()
        self.para_init()
        # self.f = open('/home/ytc/GCN/GCNcode/HGCN/pygcn/attention.txt', 'w')

        self.attention = True
        # self.lower_attention = True
        self.embedding = False


        self.write_emb = False
        if self.write_emb:
            self.emb = None

        self.nonlinear = F.relu_

        self.nclass = nclass
        self.ntype = len(nfeat_list)
        if self.embedding:
            self.mlp = nn.ModuleList()
            n_in = [nhid for _ in range(self.ntype)]
            for t in range(self.ntype):
                self.mlp.append( MLP(nfeat_list[t], n_in[t]) )
        else:
            n_in = nfeat_list

        # dim_1st = 1000
        # dim_2nd = nhid
        dim_1st = nhid
        dim_2nd = nclass + 2
        
        self.gc2 = nn.ModuleList()
        if not self.lower_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append( GraphConvolution(n_in[t], dim_1st, bias=False) )
                self.bias1 = Parameter( torch.FloatTensor(dim_1st) )
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(n_in, dim_1st, gamma=0.1)
        self.gc2.append( GraphConvolution(dim_1st, dim_2nd, bias=True) )
        # self.gc2 = GraphAttentionConvolution([dim_1st] * self.ntype, dim_2nd)

        if self.attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append( SelfAttention(dim_1st, t, 50) )
                self.at2.append( SelfAttention(dim_2nd, t, 50) )
           
        # self.outlayer = torch.nn.Linear(dim_2nd, nclass)

        self.dropout = dropout

    def para_init(self):
        self.attention = False
        self.embedding = False
        self.lower_attention = False
        self.write_emb = False

    def forward(self, x_list, adj_list, adj_all = None):
        if self.embedding:
            x0 = [None for _ in range(self.ntype)]
            for t in range(self.ntype):
                x0[t] = self.mlp[t](x_list[t])
        else:
            x0 = x_list
        
        if not self.lower_attention:
            x1 = [None for _ in range(self.ntype)]
            # 第一层gcn，与第一层后的dropout
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    if adj_list[t1][t2] is None:
                        x_t1.append
                    idx = t2
                    x_t1.append( self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1 )
                if self.attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)
                    
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )
                    # if t1 == 0:
                        # self.f.write('{0}\t{1}\t{2}\n'.format(weights[0][0].item(), weights[0][1].item(), weights[0][2].item()))
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]        
        
        x2 = [None for _ in range(self.ntype)]
        # 第二层gcn，与第二层后的softmax
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                idx = 0
                x_t1.append( self.gc2[idx](x1[t2], adj_list[t1][t2]) )
            if self.attention:
                x_t1, weights = self.at2[t1]( torch.stack(x_t1, dim=1) )
            else:
                x_t1 = reduce(torch.add, x_t1)

            # x_t1 = self.nonlinear(x_t1 / self.ntype)
            x2[t1] = x_t1
            if self.write_emb and t1 == 0:
                self.emb = F.softmax(x2[t1])
    # 单分类
            x2[t1] = F.log_softmax(x_t1, dim=1)
    # # 多分类
    #         x2[t1] = torch.sigmoid(x_t1)

        return x2

        
    def inference(self, x_list, adj_list, adj_all = None):
        return self.forward(x_list, adj_list, adj_all)

        
# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=1):
#         """Dense version of GAT."""`
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)