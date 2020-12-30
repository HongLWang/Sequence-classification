#coding=utf-8
# @Time : 20-12-24下午3:13 
# @Author : Honglian WANG

import os, sys
import argparse
parser = argparse.ArgumentParser()


# data location

realpath = sys.path[0]
parser.add_argument('--zip_path', type=str, default= realpath +
                                                     '/data/uWaveGestureLibrary.zip')
parser.add_argument('--file_path', type=str, default= realpath + '/data/time_series')
parser.add_argument('--unzip_path', type=str, default= realpath + '/data/uWaveGestureLibrary')

#attention parameter
parser.add_argument('--n_head', type=int, default=4) # number of attention multi-head
parser.add_argument('--att_drop', type=int, default=0.1)

# lstm parameter
parser.add_argument('--rnn_h_dim', type=int, default=64) # lstm hidden dimension
parser.add_argument('--att_dim', type=int, default=64) # attention embeding dimension

# training parameter
parser.add_argument('--ld_reg', type=float, default=0.001)
parser.add_argument('--lambda_l2_reg', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=0.0015)
parser.add_argument('--EPOCH', type=int, default=100)
parser.add_argument('--mini_batch_size', type=int, default=128)


#testing parameter
parser.add_argument('--train', type=float, default=0.2)
parser.add_argument('--test', type=float, default=0.8)


config = parser.parse_args()
