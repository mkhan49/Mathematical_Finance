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
Importing libraries and functions
"""
import numpy as np
from numba import jit
jit(nopython=True) # To enhance the speed of the algorithm

from thomas import *

"""
main function to call the function
"""

if __name__ == '__main__':

    N = 10
    alpha = 2 * np.ones(N) # MAIN_DIAG
    beta = np.ones(N) # SUPER_DIAG
    gamma = np.ones(N) # SUB_DIAG
    b = np.full((N,1),10)
    
    ques = tridiagonal(alpha,beta,gamma, N) # Creation of Tridiagonal matrix    
    soln = thomas(alpha, beta, gamma, b, N) # Solution obtained by the Thomas algorithm

    print("\n The tridiagonal  matrix created is:\n")
    print(ques)
    print('\n The solution of the matrix by using thomas algorithm is:\n')
    print(soln)
