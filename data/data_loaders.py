import torch
import pandas as pd
import numpy as np

import gpytorch

class wwine_qual_data():
    
    def __init__(self):
        self.df = pd.read_csv('winequality-white.csv',sep=';')
        self.data = torch.tensor(self.df.values)
        
        self.features, self.responses = self.df.columns[:-1].to_list(), self.df.columns[-1:].to_list()
        
        self.X, self.y = self.data[:,:-1], self.data[:,-1]

class rwine_qual_data():
    
    def __init__(self):
        self.df = pd.read_csv('winequality-red.csv',sep=';')
        self.data = torch.tensor(self.df.values)
        
        self.features, self.responses = self.df.columns[:-1].to_list(), self.df.columns[-1:].to_list()
        
        self.X, self.y = self.data[:,:-1], self.data[:,-1]
        
class energy_data():
    
    def __init__(self):
        self.df = pd.read_csv('energydata_complete.csv')
        self.data = self.df.values
        
        self.features, self.responses = self.df.columns[3:].to_list(), self.df.columns[1:3].to_list()
        
        self.X, self.y = self.data[:, 3:], self.data[:, 1:3]
        
