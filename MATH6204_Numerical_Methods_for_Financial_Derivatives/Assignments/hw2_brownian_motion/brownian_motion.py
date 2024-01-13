#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Importing division and print_function for compatability with python 2
"""

from __future__ import division
from __future__ import print_function

"""
                             MATH 6205 - Numerial Methods for Financial Derivatives
                                                  Fall 2018


Purpose             : The objective of this Python program is to simluate the simulate the path of Stock price using 
                      Geometric Brownian Motion as the underlying diffusion process for the Stock price. Since Weiner 
                      process is a continuous process, we used Euler discretization to discretize it and used it to 
                      simulate.The input parameters required are provided by the user.
         
         
Numerical Methods   : Geometric Brownian Motion is used for the simulation of sample paths. Euler discretization is 
                      used as a discretization method to discretize the SDE of the stock process. 
    
    
Author              : Mohammed Ameer Khan

Date of Completion  : 17 September, 2018

Files Included      : Python program file and Console output screenshot file
"""

"""
Importing numpy, scipy, and pandas libraries

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Setting the seed for the random number generation
"""
np.random.seed(seed=123)

def sample_path(S0,mu,q,sigma,T):
    ''' Simulates the path for Stock price.

    Parameters
    ==========
    S0 : float
        stock price at time 0
    T : float
        maturity date
    mu : float
        constant, stock's expected return
    q : float
        constant, time-continuous dividend yield
    sigma : float
            volatility
                
    Returns
    =======
    A 2-dimensional graph of five sample paths
    '''    
    
    NRepl = 5
    NSteps = 1000
    dt = T/NSteps
    St = np.zeros((NRepl,NSteps+1))
    St[:,0] = S0
    for i in range(NRepl):
        for j in range(NSteps):
            St[i,j+1] = St[i,j] + (mu-q) * St[i,j] * dt + sigma * St[i,j] * dt**0.5 * np.random.randn()
            
        
    x_data = pd.DataFrame(St)
    x_data = x_data.T
    plt.plot(np.linspace(0,1,1001), x_data)
    plt.title('Simulatuion of Sample Paths')
    plt.xlabel('time(t)')
    plt.ylabel('S(t)')
    plt.grid()
    plt.show()   
            

            
            
            
    
    