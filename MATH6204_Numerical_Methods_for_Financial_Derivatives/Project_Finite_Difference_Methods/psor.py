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
                      PSOR algorithm. PSOR algorithm is an extension of Gauss-Seidel alogithm which is an extension 
                      of Jacobi algorithm. It is an iterative method for solving linear system of equations. PSOR 
                      algorithm is very useful is pricing American style options. We use a relaxation parameter 
                      and a convergence condition so that the iterative solution converges. This algorithm can be 
                      used to solve any general tridiagonal matrix. We have also used numba's jit to enhance the 
                      speed of the algorithm.
                         
Numerical Methods   : Solving Tridiagonal system of equations, PSOR(Projected Successive Over Relaxation) Algorithm
                                           
Author              : Mohammed Ameer Khan

Date of Completion  : 9 December, 2018

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
Function to calculate the solution of tridiagonal system using PSOR algorithm
"""

def psor(a,b,M,N,accuracy):
    
    s = np.ones(N)
    s_old = np.ones(N)
    
    iter_counter = 0 #iteration counter
    stability_ind = 1
    w = 1.1  #relaxation parameter

    while (iter_counter == 0 or stability_ind == 1) : # loop for the iteration to calculate the solution
        
        stability_ind = 0
        iter_counter += 1
        s_new = np.zeros(N)
        
        for i in range(N):
            
            s_old[i] = s[i]
            
        for j in range(N):
            
            for k in range(j):
                
                s_new[j] += (-1*M[j][k]*s[k])
                
            for l in range(N-1,j-1,-1):
                
                s_new[j] += (-1*M[j][l] * s_old[l])
                
            s_new[j] += b[j]
            s[j] = np.maximum(a[j],s_old[j] + w/M[j][j]*s_new[j])
            
            if abs(s[j]-s_old[j])>10**(-accuracy): # accuracy condition for convergence
                stability_ind = 1
    return s