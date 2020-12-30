#coding=utf-8
# @Time : 20-12-24上午9:58 
# @Author : Honglian WANG

import os
import sys
import pandas as pd
import numpy as np

import patoolib
import rarfile
import zipfile
from os import listdir
from os.path import join
from config import config
import torch
from torch.nn.utils.rnn import pad_sequence

def remove_useless(fp): # some unrared files are not needed, eg, 0.txt
    filenames = [f for f in listdir(fp)]
    for item in filenames:
        if not 'Template' in item:
            os.remove(join(fp,item))


def extract_file():

    print ('estracting data...')

    if not os.path.exists(config.file_path):
        os.makedirs(config.file_path)

    with zipfile.ZipFile(config.zip_path, 'r') as zip_ref:
        zip_ref.extractall(config.unzip_path)

    rarfiles = [f for f in listdir(config.unzip_path) if rarfile.is_rarfile(join(config.unzip_path, f))]
    for file in rarfiles:
        filename = file[:-4]
        print(filename)
        filepath = join(config.file_path, filename)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        patoolib.extract_archive(join(config.unzip_path, file), outdir=filepath)
        remove_useless(filepath)


def load_data(fp):

    try:
        folders = os.listdir(config.file_path)
    except:
        extract_file()

    print ('loading data...')
    folders = os.listdir(config.file_path)
    data_length = []
    DATA = []
    Label = []
    Mask = []

    for fold in folders:
        fold_fp = join(config.file_path, fold)
        filenames = [f for f in listdir(fold_fp)]
        for item in filenames:
            each = item.strip().split('_')
            label = int(each[-1].split('-')[0][-1])-1

            data = pd.read_table(join(fold_fp, item), header=None, sep= ' ', index_col=None).to_numpy()

            DATA.append(torch.from_numpy(data))
            Label.append(label)
            data_length.append(data.shape[0])
            Mask.append(torch.zeros((data.shape[0], 1)))


    Padded = pad_sequence(DATA, batch_first=True)
    Mask = pad_sequence(Mask, batch_first=True, padding_value=1)

    # paddle 0 to the end and transfor all data to the same length
    # A better way to do this is to group the data by length to save space.
    # but the dataset is not big, so I am not implementing this now.
    temp = 'debug here'
    return Padded, Mask, np.array(data_length), np.array(Label)
