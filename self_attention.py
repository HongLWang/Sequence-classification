#coding=utf-8
# @Time : 20-12-24下午3:01 
# @Author : Honglian WANG

import os
import sys
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


class MultiHeadedAttention(nn.Module): #
    def __init__(self, n_head, d_input, dm, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.n_head = n_head
        self.dropout = nn.Dropout(p=dropout)
        self.dm = dm  # one layer mlp transforms input to dm dimension
        assert self.dm % self.n_head == 0
        self.dq = self.dm//self.n_head
        self.dk = self.dm//self.n_head
        self.dv = self.dm//self.n_head

        self.K_mlp = torch.nn.Linear(d_input, self.dm)  # [b,n,d]-[b,n,dm]
        self.V_mlp = torch.nn.Linear(d_input, self.dm)  # [b,n,d]-[b,n,dm]
        self.Q_mlp = torch.nn.Linear(d_input, self.dm)  # [b,1,d]-[b,1,dm]
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: batch_size * 1 * embed_dim
        :param key: batch_size * seq_len * embed_dim
        :param value: equal to key
        :param mask: batch_size * 1 * seq_len
        :return: batch_size * 1 * embed_dim
        '''
        self.seq_len = key.shape[1]
        self.batch_size = key.shape[0]
        att_out = self.multi_head_attention(query, key,value,mask)
        return att_out

    def multi_head_attention(self, Q, K, V, masked=False):
        # input = [b,n,d] -> mlp [b,n,dm=h*dq] ->reshape [b,h,n,dq]
        Q1 = self.Q_mlp(Q)
        Q2 = Q1.reshape(self.batch_size, self.n_head, 1, self.dq)
        K1 = self.Q_mlp(K)
        K2 = K1.reshape(self.batch_size, self.n_head, self.seq_len, self.dk)
        V1 = self.Q_mlp(V)
        V2= V1.reshape(self.batch_size, self.n_head, self.seq_len, self.dv)

        # 1. scaled dot product
        # inp Q:  [b, h, 1, dq] K: [b, h, n, dk]
        # out score : [b, h, 1, n]
        score = torch.matmul(Q2, K2.transpose(-1, -2), ) / torch.sqrt(torch.ones(1) * self.dk).to(device)

        # 3. evaluate the mask
        # inp score: [b, h, 1, n]
        # out score: [b, h, 1, n]
        if masked is not None:
            masked = torch.transpose(masked, -1, -2)  # [b,1,n]
            masked = torch.unsqueeze(masked, 2) #[b, 1, 1, n]
            masked = masked.to(device).bool()
            score = score.masked_fill(mask=masked, value = -1e9)
            temp = 'stop here'

        # 3. evaluate the weights
        # inp score: [b, h, 1, n]
        # out W: [b, h, 1, n]
        W = self.softmax(score)

        # 4. evaluate the context vector
        # inp W: [b, h, 1, n] V: [b, h, n, dv]
        # out H: [b, h, 1, dv]
        H = torch.matmul(W, V2)

        # 5. concatenate all heads
        # inp H: [b, h, 1, dv]
        # out H: [b, 1, dv*h]
        att_out = H.reshape(self.batch_size, 1, self.dv * self.n_head)


        return att_out




if __name__ == '__main__':
    pass



