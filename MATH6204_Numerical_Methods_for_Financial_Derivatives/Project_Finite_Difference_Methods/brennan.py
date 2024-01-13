#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Importing division and print_function for compatability with python 2
"""

from __future__ import division
from __future__ import print_function

"""
                             MATH 6205 - Numerical Methods for Financial Derivatives
                                                  Fall 2018


Purpose             : The objective of this Python program is to solve a tridiagonal system of equations using the 
                      Brennan-Schwartz algorithm. Brennan-Schwartz algorithm is a simplied version of the Gaussian 
                      elimination. This algorithm involves two core substitutions, a forward loop and a backward 
                      loop in which we calcualte a multiplier and use it as required. This algorithm can be used 
                      to solve any generate tridiagonal matrix. We have also used numba's jit to enhance the speed 
                      of the algorithm.
                         
Numerical Methods   : Solving Tridiagonal system of equations, Brennan-Schwartz Algorithm
                                           
Author              : Mohammed Ameer Khan

Date of Completion  : 9 December, 2018

Files Included      : main file, Python program file, Console output screenshot file.
"""

"""
Importing libraries

"""
import numpy as np
from numba import jit


jit(nopython=True) # To enhance the speed of the algorithm

"""
Function to solve the tridiagonal matrix using Brennan-Schwartz algorithm
"""
def brennan(alpha, gamma, beta, a, b, N):
    
    alpha_hat = np.zeros(N)
    b_hat = np.zeros(N)
    alpha_hat[0]= alpha[0]
    b_hat[0]= b[0]

    #forward loop
    
    for i in range(1, N, 1):
        
        multiplier = gamma[i]/alpha_hat[i-1]
        alpha_hat[i] = alpha[i] - multiplier * beta[i-1] 
        b_hat[i] = b[i] - multiplier *b_hat[i-1]

    soln = np.zeros(N)
    soln[N-1] = np.maximum(a[N-1],b_hat[N-1]/alpha_hat[N-1])
    
    # backward loop

    for i in range(N-2, -1, -1):
        soln[i] = np.maximum(a[i],(b_hat[i]-beta[i]* soln[i+1])/alpha_hat[i])

    return soln