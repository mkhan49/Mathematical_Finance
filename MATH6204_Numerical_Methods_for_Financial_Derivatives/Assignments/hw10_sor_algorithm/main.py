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
Importing libraries and functions
"""
import numpy as np
from numba import jit
jit(nopython=True) # To enhance the speed of the algorithm

from sor import *

"""
main function to call the function
"""

if __name__ == '__main__':

    N =10
    alpha = 2 * np.ones(N)
    beta = np.ones(N)
    gamma = np.ones(N)
    b = np.full((N,1),10)
    
    M = tridiagonal(alpha,beta,gamma, b, N) # Creation of Tridiagonal matrix    
    soln = sor(b,M,N,10) # Solution obtained by the SOR algorithm ( accuracy paramter 10^(-10))

    print("\n The tridiagonal  matrix created is:\n")
    print(M)
    print('\n The solution of the matrix by using SOR algorithm is:\n')
    print(soln)

