# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:23:08 2022

@author: sgalee2
"""

import numpy as np
import sys
import os
import pandas
import logging
import startup
import torch
from datetime import datetime
from scipy.io import loadmat

from urllib.request import urlopen
logging.getLogger().setLevel(logging.INFO)
import zipfile

from data.paths import DATA_PATH, BASE_SEED

_ALL_REGRESSION_DATATSETS = {}
tens = torch.Tensor

def add_regression(C):
    _ALL_REGRESSION_DATATSETS.update({C.name:C})
    return C

def normalize(X):
    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    return (X - X_mean) / X_std, X_mean, X_std


class Dataset(object):
    def __init__(self, split=0, prop=0.9):
        if self.needs_download:
            self.download()

        X_raw, Y_raw = self.read_data()
        X, Y = self.preprocess_data(X_raw, Y_raw)

        ind = np.arange(self.N)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        self.X_train = X[ind[:n]]
        self.Y_train = Y[ind[:n]]

        self.X_test = X[ind[n:]]
        self.Y_test = Y[ind[n:]]

    @property
    def datadir(self):
        dir = os.path.join(DATA_PATH, self.name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir

    @property
    def datapath(self):
        filename = self.url.split('/')[-1]  # this is for the simple case with no zipped files
        return os.path.join(self.datadir, filename)

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        logging.info('donwloading {} data'.format(self.name))

        is_zipped = np.any([z in self.url for z in ['.gz', '.zip', '.tar']])

        if is_zipped:
            filename = os.path.join(self.datadir, self.url.split('/')[-1])
        else:
            filename = self.datapath

        with urlopen(self.url) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        if is_zipped:
            zip_ref = zipfile.ZipFile(filename, 'r')
            zip_ref.extractall(self.datadir)
            zip_ref.close()

            # os.remove(filename)

        logging.info('finished donwloading {} data'.format(self.name))

    def read_data(self):
        raise NotImplementedError

    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        Y, self.Y_mean, self.Y_std = normalize(Y)
        return tens(X), tens(Y)


uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


@add_regression
class Boston(Dataset):
    N, D, name = 506, 13, 'boston'
    url = uci_base_url + 'housing/housing.data'

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Concrete(Dataset):
    N, D, name = 1030, 8, 'concrete'
    url = uci_base_url + 'concrete/compressive/Concrete_Data.xls'

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Energy(Dataset):
    N, D, name = 768, 8, 'energy'
    url = uci_base_url + '00242/ENB2012_data.xlsx'
    def read_data(self):
        # NB this is the first output (aka Energy1, as opposed to Energy2)
        data = pandas.read_excel(self.datapath).values[:, :-1]
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Kin8mn(Dataset):
    N, D, name = 8192, 8, 'kin8nm'
    url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'
    def read_data(self):
        data = pandas.read_csv(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Naval(Dataset):
    N, D, name = 11934, 14, 'naval'
    url = uci_base_url + '00316/UCI%20CBM%20Dataset.zip'

    @property
    def datapath(self):
        return os.path.join(self.datadir, 'UCI CBM Dataset/data.txt')

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values
        # NB this is the first output
        X = data[:, :-2]
        Y = data[:, -2].reshape(-1, 1)

        # dims 8 and 11 have std=0:
        X = np.delete(X, [8, 11], axis=1)
        return X, Y


@add_regression
class Power(Dataset):
    N, D, name = 9568, 4, 'power'
    url = uci_base_url + '00294/CCPP.zip'

    @property
    def datapath(self):
        return os.path.join(self.datadir, 'CCPP/Folds5x2_pp.xlsx')

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Protein(Dataset):
    N, D, name = 45730, 9, 'protein'
    url = uci_base_url + '00265/CASP.csv'

    def read_data(self):
        data = pandas.read_csv(self.datapath).values
        return data[:, 1:], data[:, 0].reshape(-1, 1)


@add_regression
class WineRed(Dataset):
    N, D, name = 1599, 11, 'winered'
    url = uci_base_url + 'wine-quality/winequality-red.csv'

    def read_data(self):
        data = pandas.read_csv(self.datapath, delimiter=';').values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class WineWhite(WineRed):
    N, D, name = 4898, 11, 'winewhite'
    url = uci_base_url + 'wine-quality/winequality-white.csv'


@add_regression
class Yacht(Dataset):
    N, D, name = 308, 6, 'yacht'
    url = uci_base_url + '/00243/yacht_hydrodynamics.data'

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values[:-1, :]
        return data[:, :-1], data[:, -1].reshape(-1, 1)
    
regression_datasets = list(_ALL_REGRESSION_DATATSETS.keys())
regression_datasets.sort()

def get_regression_data(name, *args, **kwargs):
    return _ALL_REGRESSION_DATATSETS[name](*args, **kwargs)
