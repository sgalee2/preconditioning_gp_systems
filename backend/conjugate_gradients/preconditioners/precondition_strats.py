# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:37 2022

@author: adayr
"""
from backend.conjugate_gradients.utils import common_terms

class Preconditioner:
    
    def mvp(self, v):
        raise NotImplementedError
        
    def imvp(self, v):
        raise NotImplementedError()
        
