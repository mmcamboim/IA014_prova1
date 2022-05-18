# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:29:35 2022

@author: mcamboim
"""

import numpy as np

class generic_dynamic_model():
    
    def __init__(self, A : np.array, B : np.array, C : np.array, D : np.array):
        self.__x = np.zeros(2).reshape(-1,1)
        self.__y = np.zeros(2).reshape(-1,1)
        self.__A = A
        self.__B = B
        self.__C = C
        self.__D = D
    
    def iteration(self, exogenous_input : list):
        u = np.array(exogenous_input).reshape(2,1)
        self.__y = self.__C @ self.__x + self.__D @ u
        self.__x = self.__A @ self.__x + self.__B @ u
        
    @property
    def output(self):
        return self.__y