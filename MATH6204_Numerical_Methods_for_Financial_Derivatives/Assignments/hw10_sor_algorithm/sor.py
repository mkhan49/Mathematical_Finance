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
                      SOR algorithm. SOR algorithm is an extension of Gauss-Seidel alogithm which is an extension 
                      of Jacobi algorithm. It is an iterative method for solving linear system of equations. SOR 
                      algorithm is very useful is pricing American style options. We use a relaxation parameter 
                      and a convergence condition so that the iterative solution converges. This algorithm can be 
                      used to solve any general tridiagonal matrix. We have also used numba's jit to enhance the 
                      speed of the Thomas algorithm.
                         
Numerical Methods   : Solving Tridiagonal system of equations, SOR(Successive Over Relaxation) Algorithm
                                           
Author              : Mohammed Ameer Khan

Date of Completion  : 30 November, 2018

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

def tridiagonal(alpha,gamma,beta, b, N):
    
    system = np.array([alpha,gamma,beta])
    tri_init =np.array([0,-1,1])
    tri_diagonal = spdiags(system,tri_init,N,N).toarray()
    return tri_diagonal 

"""
Function to find the relaxation parameter in order to use it in sor algorithm
"""

def relaxation_parameter(M,N):
    
    inv_mat = np.zeros((N,N))
    low_upp_mat = np.zeros((N,N))
    ele_mat = np.zeros(N)
    relax_parameter = 0
    
    for i in range(N):
        for j in range(N):
            
            if i==j:
                
                inv_mat[i][j] = 1/M[i][j]
                low_upp_mat[i][j] = 0
                
            else:
                
                inv_mat[i][j] = 0
                low_upp_mat[i][j] = -1 * M[i][j]
                
    mult_mat = np.matmul(inv_mat, low_upp_mat)
    eig_val = np.linalg.eigvals(mult_mat)  # Eigen values calculation
    
    for k in range(N):
        
        ele_mat[k] = 2.0/(1+np.sqrt(1-eig_val[k]**2))
        
        if (ele_mat[k]>=1 and ele_mat[k]<2 and ele_mat[k]>relax_parameter) or k==0:
            
            relax_parameter = ele_mat[k]
            
    return relax_parameter

"""
Function to calculate the solution of tridiagonal system using SOR algorithm
"""

def sor(b,M,N,accuracy):
    
    s = np.ones(N)
    s_old = np.ones(N)
    
    iter_counter = 0 #iteration counter
    stability_ind = 1
    w = relaxation_parameter(M,N)

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
            s[j] = s_old[j] + w/M[j][j]*s_new[j]
            
            if abs(s[j]-s_old[j])>10**(-accuracy): # accuracy condition for convergence
                stability_ind = 1
    return s