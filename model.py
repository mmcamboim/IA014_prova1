# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:55:58 2022

@author: mcamboim
"""
import numpy as np

def dynamic_model() -> tuple:
    A = np.array([0.95,0.1,-0.1,0.8]).reshape(2,2)
    B = np.array([2.0,0.01,0.05,0.5]).reshape(2,2)
    C = np.array([1,0.0,0.5,1.0]).reshape(2,2)
    D = np.array([1.5,0.05,0.01,0.8]).reshape(2,2)
    
    return A,B,C,D
