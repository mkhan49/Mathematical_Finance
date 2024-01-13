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


Purpose             : The objective of this Python program is to compute the values of American Call and Put options 
                      using Monte Carlo simulation. Geometric Brownian Motion as the underlying diffusion process for 
                      the Stock price. Since Weiner process is a continuous process, we used Euler discretization to 
                      discretize it and used it to simulate. In order to calculate the prices of American style of 
                      options, we need to calculate the discounted risk neutral expectations. Also, American options
                      can be exercised at any point of time so we need to calculate the optimal stopping time as well
                      in order to compute the discounted expectation. The Monte Carlo simulation along with the 
                      regression technique can be used to compute the prices of American options. A linear regression
                      is applied at each stopping time which in our case is the time step of the discretization. The
                      independent variable (x) will be our stock values generated using the Euler discretization and 
                      the dependent varible (y) will be the discounted payoff values from the immediate next time step.
                      This process is applied to all the time steps and then the average of initial time step (t0) gives
                      us the price of the American option at time, t0. We have used a third order polynomial for the 
                      regression and the Least Squares method is used to find the optimal coefficients.
         
         
Numerical Methods   : Geometric Brownian Motion is used for the simulation of sample paths. Euler discretization is used 
                      as a discretization method to discretize the SDE of the stock process. Monte Carlo simulations is 
                      used for the valuation of American options. Regression is used at each stopping times and Least Squares
                      is to optimize the regression coefficients.
    
    
Author              : Mohammed Ameer Khan

Date of Completion  : 18 November, 2018

Files Included      : main file, Python program file, Console output screenshot file and analysis document.
"""


"""
Importing functions
"""

from mc_american_options import *

"""
main function to call the function
"""

if __name__ == '__main__':
    print('\nMonte Carlo Regression method I values of American Call\n')
    print('Time step', ' MC_value')
    monte_carlo_reg(100,100,0.03,0.025,0.75,1,0.01,1)
    monte_carlo_reg(100,100,0.03,0.025,0.75,1,0.001,1)
    
    print('\nMonte Carlo Regression method I values of American Put\n')
    print('Time step', ' MC_value')
    monte_carlo_reg(100,100,0.03,0.025,0.75,1,0.01,0) 
    monte_carlo_reg(100,100,0.03,0.025,0.75,1,0.001,0)
