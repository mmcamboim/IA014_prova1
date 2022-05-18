# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:55:33 2022

@author: mcamboim
"""
import numpy as np
import matplotlib.pyplot as plt

from model import dynamic_model
from generic_model import generic_dynamic_model
from rho_kalman import rho_kalman

plt.close('all')

plt.rcParams['axes.linewidth'] = 2.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

steps = 100

# Load Model -----------------------------------------------------------------
A,B,C,D = dynamic_model()

# Unit Impulse Response ------------------------------------------------------
# First Column: Unit Impulse Response Determination
model = generic_dynamic_model(A,B,C,D)
unit_impulse_1 = np.zeros((steps,2))
unit_impulse_1[0,0] = 1.0
yk_1 = np.zeros((steps,2))
for k in range(steps):
    exogenous_input = list(unit_impulse_1[k])
    model.iteration(exogenous_input)
    yk_1[k,:] = model.output.reshape(1,2)

# Second Column: Unit Impulse Response Determination
model = generic_dynamic_model(A,B,C,D)
unit_impulse_2 = np.zeros((steps,2))
unit_impulse_2[0,1] = 1.0
yk_2 = np.zeros((steps,2))

for k in range(steps):
    exogenous_input = list(unit_impulse_2[k])
    model.iteration(exogenous_input)
    yk_2[k,:] = model.output.reshape(1,2)

# Rho Kalman Method ----------------------------------------------------------
rk = rho_kalman(yk_1,yk_2)
A_hat,B_hat,C_hat,D_hat = rk.rho_kalman(49)

# Models Comparison-----------------------------------------------------------
rk_model = generic_dynamic_model(A_hat,B_hat,C_hat,D_hat)
model = generic_dynamic_model(A,B,C,D)

yk_model = np.zeros((steps,2))
yk_rk = np.zeros((steps,2))
for k in range(steps):
    exogenous_input = [np.sin(k/10),np.cos(k/10)]
    rk_model.iteration(exogenous_input)
    model.iteration(exogenous_input)
    yk_model[k,:] = model.output.reshape(1,2)
    yk_rk[k,:] = rk_model.output.reshape(1,2)

# Plots ----------------------------------------------------------------------
# Unit Impulse Response
plt.figure(figsize=(6,6),dpi=150)
plt.subplot(2,1,1)
plt.plot(yk_1[:,0],c='b',lw=3)
plt.plot(yk_1[:,1],c='r',lw=3)
plt.legend(['$h_1(k)$','$h_2(k)$'])
plt.xlabel('(a)')
plt.ylabel('Saídas []')
plt.grid(True,ls='dotted')

plt.subplot(2,1,2)
plt.plot(yk_2[:,0],c='b',lw=3)
plt.plot(yk_2[:,1],c='r',lw=3)
plt.legend(['$h_1(k)$','$h_2(k)$'])
plt.xlabel('Passo [k]\n(b)')
plt.ylabel('Saídas []')
plt.grid(True,ls='dotted')

plt.tight_layout()

# Outputs Comparison
# Output 1
plt.figure(figsize=(6,6),dpi=150)
plt.subplot(2,1,1)
plt.plot(yk_model[:,0],c='b',lw=3)
plt.plot(yk_rk[:,0],c='r',lw=3)
plt.legend(['$y_1(k)$: Modelo','$y_1(k)$: Ho-Kalman'])
plt.xlabel('(a)')
plt.ylabel('Saídas []')
plt.ylim([-40,25])
plt.grid(True,ls='dotted')

plt.subplot(2,1,2)
plt.plot(yk_model[:,0] - yk_rk[:,0],c='b',lw=3)
plt.xlabel('Passo [k]\n(b)')
plt.ylabel('Erro []')
plt.ylim([-1e-12,1e-12])
plt.grid(True,ls='dotted')
plt.tight_layout()

# Output 2
plt.figure(figsize=(6,6),dpi=150)
plt.subplot(2,1,1)
plt.plot(yk_model[:,1],c='b',lw=3)
plt.plot(yk_rk[:,1],c='r',lw=3)
plt.legend(['$y_2(k)$: Modelo','$y_2(k)$: Ho-Kalman'])
plt.xlabel('(a)')
plt.ylabel('Saídas []')
plt.ylim([-15,10])
plt.grid(True,ls='dotted')

plt.subplot(2,1,2)
plt.plot(yk_model[:,1] - yk_rk[:,1],c='b',lw=3)
plt.xlabel('(b)\nPasso [k]')
plt.ylabel('Erro []')
plt.ylim([-1e-12,1e-12])
plt.grid(True,ls='dotted')
plt.tight_layout()