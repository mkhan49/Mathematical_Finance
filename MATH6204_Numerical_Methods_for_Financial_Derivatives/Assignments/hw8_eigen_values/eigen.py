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


Purpose             : The objective of this Python program is to compute the solution of a tridiagonal matrix using 
                      the finite differences method. WE use the heat equation to solve for explicit or implicit
                      solution. Based on the given solution, we can discretize the linear system of difference 
                      euations at given point of time using the finite difference method. The lambda parameter
                      is given by the user. prices of European calls and puts using  

         
Numerical Methods   : Heat equation is used to solve for explicit or implicit solution. Solving Tridiagonal system 
                      of equations. Finite Difference method to solve linear system of equations is used. 
                      
Author              : Mohammed Ameer Khan

Date of Completion  : 2 December, 2018

Files Included      : main file, Python program file, Console output screenshot file.
"""

"""
Importing libraries and functions
"""
import numpy as np
from scipy.sparse import spdiags
from math import sqrt, cos, pi

"""
Function to create a tridiagonal matrix
"""

def tridiagonal(alpha,gamma,beta, N):
    
    data = np.array([alpha,gamma,beta])
    triset =np.array([0,-1,1])
    tridiagonal = spdiags(data,triset,N-1,N-1).toarray()
    
    return tridiagonal

"""
Function to create a tridiagonal matrix
"""

def eigen_diagonal(alpha, beta, gamma, k):
    
	eigen = np.zeros(k)
	for i in range(k):
		eigen[i] = alpha + 2*beta*sqrt(gamma/beta)*cos((i+1)*pi/(k+1))
	return eigen

"""
Function to create the eigen values
"""


def eigen_value(alpha, gamma, beta, lamda, M, N, Ind ):
    ''' Calculates the eigen values for stability analysis.

    Parameters
    ==========
    alpha : array
            main diagonal of matrix
    gamma : array
           super diagonal of matrix
    beta  : array
           sub diagonal of matrix
    lamda : scalar
    M : matrix
    
    Ind : integer(Indicator)
                1 - corresponds to Explicit solution
                0 - corresponds to Implicit solution
    '''

    if Ind== 0:

        eigen1 = sorted(np.linalg.eigvals(np.linalg.inv(M)))
        eigen2 = 1.0 + lamda * eigen_diagonal(2-alpha[0], -beta[0], -gamma[0], N-1)
        eigen2 = sorted(1.0/eigen2)
        
        if np.allclose(eigen1,eigen2):
            return eigen1
        else:
            print('No solution because eigen values are unstable')

            
    elif Ind == 1:
        
        eigen1 = sorted(np.linalg.eigvals(M))
        eigen2 = sorted(eigen_diagonal(alpha[0], beta[0], gamma[0], N-1))
        
        if np.allclose(eigen1,eigen2):
            return eigen1
        else:
            print('No solution because eigen values are unstable')
        
    return eigen1
		


























