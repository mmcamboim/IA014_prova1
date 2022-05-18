# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:25:09 2022

@author: mcamboim
"""
import numpy as np

def left_pinv(A : np.array) -> np.array: 
    return np.linalg.inv(A.T @ A) @ A.T

def right_pinv(A : np.array) -> np.array:
    return A.T @ np.linalg.inv(A @ A.T)

class rho_kalman():
    
    def __init__(self,gk_1 : np.array, gk_2: np.array):
        self.__gk_1 = gk_1
        self.__gk_2 = gk_2
        self.__n_inputs = 2
        self.__n_outputs = 2
        
    def rho_kalman(self, lines_columns : float) -> np.array:
        H,H_up = self.__hankel(1,lines_columns) 
        observability,controllability = self.__hankel_decomposition(H)
        A_hat,B_hat,C_hat,D_hat = self.__find_SSM(H,H_up,observability,controllability)
        return A_hat,B_hat,C_hat,D_hat
    
    def __hankel(self, init_step : float, lines_columns : float) -> tuple:
        H_temp = np.zeros(((lines_columns+1)*self.__n_inputs,lines_columns*self.__n_outputs))
        for line in range(lines_columns+1):
            for column in range(lines_columns):
                start_line_idx = 2*line
                stop_line_idx =2*line+2
                start_column_idx = 2*column
                stop_column_idx = 2*column + 2
                step = init_step + column + line
                
                H_temp[start_line_idx:stop_line_idx,start_column_idx:stop_column_idx] = self.get_impulse_response(step)
        H = H_temp[:-self.__n_outputs,:]
        H_up = H_temp[self.__n_outputs:,:]
        return H,H_up
    
    def get_impulse_response(self, step : float) -> np.array:
        G = np.zeros((2,2))
        G[:,0] = self.__gk_1[step,:]
        G[:,1] = self.__gk_2[step,:]
        return G
        
    def __hankel_decomposition(self, H : np.array) -> tuple:
        U,S,VT = np.linalg.svd(H)
        meaningful_values = np.sum(S > 1e-5) 
        U = U[:,0:meaningful_values]
        S = np.diag(S[0:meaningful_values])
        VT = VT[0:meaningful_values,:]
        
        observability = U @ np.sqrt(S)
        controllability = np.sqrt(S) @ VT
        
        return observability,controllability
    
    def __find_SSM(self,H : np.array, H_up : np.array, observability : np.array, controllability : np.array) -> tuple:
        A_hat = self.__find_A(H_up,observability,controllability)
        B_hat = self.__find_B(H,observability)
        C_hat = self.__find_C(H,controllability)
        D_hat = self.__find_D()
        
        return A_hat,B_hat,C_hat,D_hat
    
    def __find_A(self,H_up : np.array, observability : np.array, controllability : np.array) -> np.array:
        observability_lpinv = left_pinv(observability)
        controllability_rpinv = right_pinv(controllability)
        A_hat = observability_lpinv @ H_up @ controllability_rpinv
        return A_hat
    
    def __find_B(self, H : np.array, observability : np.array) -> np.array:
        observability_lpinv = left_pinv(observability)
        B_hat = observability_lpinv @ H[:,0:self.__n_inputs]
        return B_hat
        
    def __find_C(self, H : np.array, controllability : np.array) -> np.array:
        controllability_rpinv = right_pinv(controllability)
        C_hat = H[0:self.__n_outputs,:] @ controllability_rpinv
        return C_hat
    
    def __find_D(self) -> np.array:
        D_hat = self.get_impulse_response(0)
        return D_hat
    
    
    
    

    