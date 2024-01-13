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
                      Thomas algorithm. Thomas algorithm is a simplied version of the Gaussian elimination. This 
                      algorithm involves two core substitutions, a forward loop and a backward loop in which we 
                      calcualte a multiplier and use it as required. This algorithm can be used to solve any general
                      tridiagonal matrix. We have also used numba's jit to enhance the speed of the Thomas algorithm.
                         
Numerical Methods   : Solving Tridiagonal system of equations, Thomas Algorithm(Gaussian Elimination)
                                           
Author              : Mohammed Ameer Khan

Date of Completion  : 25 November, 2018

Files Included      : main file, Python program file, Console output screenshot file.
"""

"""
Importing libraries

"""
import numpy as np
from scipy.sparse import spdiags
from numba import jit


jit(nopython=True) # To enhance the speed of the algorithm

"""
Function to create a tridiagonal matrix
"""

def tridiagonal(alpha,gamma,beta, N):
    
    system = np.array([alpha,gamma,beta])
    tri_init =np.array([0,-1,1])
    tri_diagonal = spdiags(system,tri_init,N,N).toarray()
    return tri_diagonal 


"""
Function to solve the tridiagonal matrix using Thomas algorithm
"""
def thomas(alpha, gamma, beta, b, N):
    
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
    soln[N-1] = b_hat[N-1]/alpha_hat[N-1]
    
    # backward loop

    for i in range(N-2, -1, -1):
        soln[i] = (b_hat[i]-beta[i]* soln[i+1])/alpha_hat[i]

    return soln