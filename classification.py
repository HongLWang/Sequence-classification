#coding=utf-8
# @Time : 20-12-24下午3:11 
# @Author : Honglian WANG

import os
import sys
from sklearn.metrics import classification_report

import numpy as np
from numpy import Inf
from torch import nn
import  torch
import torch.optim as optim
from self_attention import MultiHeadedAttention
from config import config
from load_dataset import load_data
from tools import mini_batch, train_test_split
import torch.nn.functional as F
import sklearn
from matplotlib import pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

class GestureClassification(torch.nn.Module):

    def __init__(self,fea_dim):
        
        super().__init__()

        self.fea_dim = fea_dim

        self.lstm = torch.nn.LSTM(fea_dim, config.rnn_h_dim,
                                  bidirectional=False, batch_first=True)

        self.attention = MultiHeadedAttention(config.n_head,
                                              config.rnn_h_dim, config.att_dim, config.att_drop)

        self.mlp1 = torch.nn.Linear(config.att_dim, 16)
        self.mlp2 = torch.nn.Linear(16, 8)
        self.softmax = nn.Softmax(dim=1)

        # batch normalization
        self.dense_bn1 = nn.BatchNorm1d(config.att_dim)
        self.dense_bn2 = nn.BatchNorm1d(16)
        self.dense_bn3 = nn.BatchNorm1d(8)

    def forward(self, x, real_len, mask):
        ''' feed data and give output'''

        hidden = self.lstm(x)[0]  # [batch_size, seq_len, embed_dim]
        query = self.tensor_indexing(hidden, real_len)  #[batch_size, 1, embed_dim]
        att_out = self.attention(query,hidden,hidden,mask)
        att_out = att_out + query # residuel connection [b,1,att_dim]
        att_out = torch.squeeze(att_out,dim=1) # [b,1,att_dim]->[b,att_dim]

        if x.shape[0] == 1:  # batch_norm requires more than 1 data on each channel
            # att_out.shape = [1, seq_len]
            x = att_out / torch.sum(att_out)
            p = self.mlp1(x)
            p = F.relu(p/torch.sum(p))
            p = self.mlp2(p)
            p = F.relu(p/torch.sum(p))
            prediction = self.softmax(p)
        else:
            x = self.dense_bn1(att_out)
            p = self.mlp1(x)
            p = F.relu(self.dense_bn2(p))
            p = self.mlp2(p)
            p = F.relu(self.dense_bn3(p))
            prediction = self.softmax(p)


        return prediction

    def tensor_indexing(self, tensor, index):

        d1, d2, d3 = tensor.shape
        index = index - 1  # index should be len of sequence -1
        index1 = index.reshape((d1, 1))
        index2 = torch.from_numpy(index1).long()
        index3 = index2.repeat(1, d3)
        index4 = torch.unsqueeze(index3, 1)
        return torch.gather(tensor, 1, index4.to(device))


def batch_training(model, train_X, train_S, train_Y, train_M, optimizer, lr_scheduler):

    acc_arr = []
    recall_arr = []
    loss_arr = []
    for epoch in range(config.EPOCH):
        total_loss = 0
        minibatch = mini_batch(train_X, train_S, train_Y, train_M)
        cnt = 0

        for (x, s, label, mask) in minibatch:
            # perform prediction
            prediction = model(x,s,mask) # [num_seq * num_label]
            task_loss = F.cross_entropy(prediction, label)
            optimizer.zero_grad()
            task_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            total_loss += task_loss.item()


            ranked_prediction = torch.argmax(prediction, dim=1)
            prediction_c = ranked_prediction.cpu().data.numpy()
            cnt += 1

            if cnt == 1:
                PRE = prediction_c
                LAB = label.cpu().data.numpy()
            if cnt > 1:
                PRE = np.concatenate((PRE,prediction_c), axis=0)
                LAB = np.concatenate((LAB,label.cpu().data.numpy()), axis=0)


        print ('total loss at epoch %d is : %f' %(epoch, total_loss))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'model/'+
                       str(epoch) + '.pkl')
        loss_arr.append(total_loss)

        acc, recall = metric(LAB, PRE)
        acc_arr.append(acc)
        recall_arr.append(recall)

    plot_acc(acc_arr, recall_arr)

def plot_acc(acc, recall):
    x = list(np.arange(len(acc)))
    l1 = plt.plot(x, acc, 'r--', label='accuracy')
    l2 = plt.plot(x, recall, 'g--', label='recall')
    plt.plot(x, acc, 'r-', x, recall, 'g-')
    plt.title('The accuracy and recall curve')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()
    plt.show()


def metric(y_true, y_pred):
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    return acc, recall


def Prediction(model, test_X, test_L, test_Y,test_M):


    with torch.no_grad():
        minibatch = mini_batch(test_X, test_L, test_Y,test_M)
        cnt = 0
        PRE = []
        LAB = []
        for (x, s, label, mask) in minibatch:

            prediction = model(x,s, mask)  # [num_seq * num_label]
            ranked_prediction = torch.argmax(prediction, dim=1)
            prediction = ranked_prediction.cpu().data.numpy()
            cnt += 1

            if cnt == 1:
                PRE = prediction
                LAB = label
            if cnt > 1:
                PRE = np.concatenate((PRE,prediction), axis=0)
                LAB = np.concatenate((LAB,label), axis=0)

        print(classification_report(LAB, PRE))




if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    Data, Mask, Seq_len, Label = load_data(config.file_path)

    print('Data.shape', Data.shape)

    fea_dim = Data.shape[-1]
    [train_X,train_L,train_Y,train_M], [test_X, test_L, test_Y,test_M] = train_test_split(Data, Seq_len, Label, Mask)

    print ('train_X.shape', train_X.shape)
    print ('testX.shape', test_X.shape)


    model = GestureClassification(fea_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.lambda_l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    train_X = train_X.to(device).float()
    train_M = train_M.to(device).long()
    train_Y = torch.tensor(train_Y, device=device)

    batch_training(model, train_X, train_L, train_Y, train_M, optimizer, scheduler)

    test_X = test_X.to(device).float()
    test_M = test_M.to(device).long()

    # model = GestureClassification(fea_dim)
    # model.load_state_dict(torch.load('model/90.pkl'))
    # model.to(device)
    Prediction(model, test_X, test_L, test_Y, test_M)
