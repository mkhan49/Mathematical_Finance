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
import matplotlib.pyplot as plt

from eigen import *

"""
main function to call the function
"""

if __name__ == '__main__':
    
    #Defining parameters
    
    N =10
    dx = np.array([0.005,0.05,0.04,0.04])
    dt = np.array([0.001,0.0015,0.001,0.0015])
    dx2= dx**2
    lamda= dt/dx2
   
    
    for i in range(len(lamda)):
        
        #Definiing the elements of the matrix
        alpha = np.ones(N-1)*(1-2*lamda[i])
        beta = np.ones(N-1)*lamda[i]
        gamma = np.ones(N-1)*lamda[i]
        
        #Defining the matrix for implicit and explicit solution
        Mexpl = tridiagonal(alpha,beta,gamma, N)
        Mimpl= tridiagonal(-alpha+2,-beta,-gamma, N)
        eigen_expl=eigen_value(alpha, gamma, beta, lamda[i], Mimpl, N, 0)
        eigen_impl=eigen_value(alpha, gamma, beta, lamda[i], Mexpl, N, 1)

        option_fig = plt.figure()
        plt.suptitle("Eigen values of \lambda: {0}".format(lamda[i]),fontsize=9)

        # Plots of implict and explicit solution for call and put options.
        call_fig = option_fig.add_subplot(211)
        call_fig.plot(eigen_impl,color="purple")
        plt.axhline(y=1, ls=':')
        plt.axhline(y=-1, ls=':')
        put_fig = option_fig.add_subplot(212)
        put_fig.plot(eigen_expl,color="orange")   
        plt.axhline(y=1, ls=':')
        plt.axhline(y=-1, ls=':')
	
        option_fig.subplots_adjust(hspace=0.5)
        plt.show()













