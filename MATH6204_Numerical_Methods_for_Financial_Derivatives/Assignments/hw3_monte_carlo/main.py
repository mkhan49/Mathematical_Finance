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


Purpose             : The objective of this Python program is to simluate the simulate the path of Stock price using 
                      Geometric Brownian Motion as the underlying diffusion process for the Stock price. Since Weiner 
                      process is a continuous process, we used Euler discretization and Milstein discretization to 
                      discretize it and used it to simulate. Monte Carlo simulations is used to calculate the 
                      European call and put option values. The error is calculated between the Monte Carlo price and the 
                      Black-Scholes price (analytical solution). The input parameters required are provided by the user.
         
         
Numerical Methods   : Geometric Brownian Motion is used for the simulation of sample paths. Euler discretization and 
                      Milstein discretization are used as a discretization method to discretize the SDE of the stock 
                      process. Monte Carlo simulations and Black Scholes formuale is used to calculate the option values.
    
    
Author              : Mohammed Ameer Khan

Date of Completion  : 25 September, 2018

Files Included      : main file, Python program file, Console output screenshot file and analysis document.
"""


"""
Importing functions

"""
from hw3_monte_carlo import *

"""
main function to call the function
"""

if __name__ == '__main__':
    print('\nMonte Carlo and Black Scholes option values for European Call\n')
    print('Time step', ' MC_value','   BSM_Value', '  Abs diff', 'Disc method \n')
    monte_carlo_euler(100,100,0.03,0.025,0.75,1,0.01,1)  
    monte_carlo_euler(100,100,0.03,0.025,0.75,1,0.001,1)
    monte_carlo_milstein(100,100,0.03,0.025,0.75,1,0.01,1)  
    monte_carlo_milstein(100,100,0.03,0.025,0.75,1,0.001,1)

    print('\nMonte Carlo and Black Scholes option values for European Put\n')
    print('Time step', ' MC_value','   BSM_Value', '  Abs diff', 'Disc method \n')
    monte_carlo_euler(100,100,0.03,0.025,0.75,1,0.01,0)  
    monte_carlo_euler(100,100,0.03,0.025,0.75,1,0.001,0)
    monte_carlo_milstein(100,100,0.03,0.025,0.75,1,0.01,0)  
    monte_carlo_milstein(100,100,0.03,0.025,0.75,1,0.001,0)
