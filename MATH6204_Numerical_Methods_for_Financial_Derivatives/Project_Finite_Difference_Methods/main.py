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


Purpose             : The objective of this Python program is to compute the prices of European and American calls and puts 
                      using Finite Difference Methods(FDMs), Monte Carlo Simulations and the closed form solutions. Every
                      method has its pros and cons. We have used Explicit, Implicit and Crank-Nicholson discretization
                      methods to solve using differing algorithms such as Thomas, Brennan-Schwartz, SOR and PSOR. We have
                      seen that American options are costlier than European counterparts using different methods. Also, we 
                      have calculated the absolute difference compared to the closed form, Black Scholes solution. Monte Carlo
                      simulations were performed to price European options and also regression based method II in Monte Carlo
                      simulation to price American options. 
         
Numerical Methods   : Geometric Brownian Motion is used for the simulation of sample paths. Euler discretization is used 
                      as a discretization method to discretize the SDE of the stock process. Monte Carlo simulations and 
                      Black Scholes formuale is used to calculate the option values.Monte Carlo simulations is also
                      used for the valuation of American options. Regression is used at each stopping times and Least Squares
                      is to optimize the regression coefficients. Finite Difference Methods(FDMs) are used to discretize the 
                      heat equation. Explicit Method, Implicit Method and Crank-Nicholson Methods of FDM are used. Solving 
                      Tridiagonal system of equations, Thomas Algorithm(Gaussian Elimination).SOR(Successive Over Relaxation) 
                      Algorithm. Brennan-Schwartz Algorithm. PSOR(Projected Successuve Over Relaxation) Algorithm.
                      
Author              : Mohammed Ameer Khan

Date of Completion  : 10 December, 2018

Files Included      : main file, Python program file, Console output screenshot file, Summary document.
"""

"""
Importing libraries and functions
"""
import numpy as np
import pandas as pd
from numba import jit
jit(nopython=True) # To enhance the speed of the algorithm

#importing required functions from other python programs

from mc_european import *
from mc_american import *
from explicit import *
from fdm import *
from thomas import *
from sor import *
from brennan import *
from psor import *

"""
main function to call the other functions
"""

if __name__ == '__main__':
    
    # Initializing parameters
    
    S0 = 100
    K  = 100 
    T  = 1
    r  = 0.02
    q = 0.01
    sigma = 0.6
    dx = 0.05
    dt = 0.00125
    dtau = 0.00125
    xmin = -2.5
    xmax = 2.5
    
    # Question 1
    
    # Calculating prices of European optins using FDMs
    
    print('\n------------------------------Question 1----------------------------------')
    
    fdm_eur = {'FDM   ':('Explicit','Implicit','Implicit','Crank-Nicholson','Crank-Nicholson'),
           'Algorithm':('Explicit','Thomas','SOR','Thomas','SOR'),
           'European Call':(expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 1, 1),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 0, 1),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 1, 1),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 0, 1),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 1, 1)),
            'Abs er_c':(abs(expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 1, 1)-BSM_value(S0, K, r, q, sigma, T, 1)),
                        abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 0, 1)-BSM_value(S0, K, r, q, sigma, T, 1)),
                         abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 1, 1)-BSM_value(S0, K, r, q, sigma, T, 1)),
                         abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 0, 1)-BSM_value(S0, K, r, q, sigma, T, 1)),
                         abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 1, 1)-BSM_value(S0, K, r, q, sigma, T, 1))),
           'European Put':(expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 0,  1),
                           fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 0, 1),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 1, 1),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 0, 1),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 1, 1)),
            'Abs er_p':(abs(expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 0, 1)-BSM_value(S0, K, r, q, sigma, T, 0)),
                        abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 0, 1)-BSM_value(S0, K, r, q, sigma, T, 0)),
                         abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 1, 1)-BSM_value(S0, K, r, q, sigma, T, 0)),
                         abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 0, 1)-BSM_value(S0, K, r, q, sigma, T, 0)),
                         abs(fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 1, 1)-BSM_value(S0, K, r, q, sigma, T, 0)))}
            
    fdm_eur_output = pd.DataFrame(data = fdm_eur)
    fdm_eur_output = fdm_eur_output[['FDM   ','Algorithm','European Call','Abs er_c','European Put','Abs er_p']] # rearranging columns
    print('\n 1(a). Finite Difference Methods (FDMs) for European Options')
    print('\n',fdm_eur_output,'\n')    
    
    # Calculating prices of American optins using FDMs

    print('----------------------------------------------------------------------------')
    
    fdm_ame = {'FDM   ':('Explicit','Implicit','Implicit','Crank-Nicholson','Crank-Nicholson'),
           'Algorithm':('Explicit','Brennan','PSOR','Brennan','PSOR'),
           'American Call':(expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 1, 0),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 0, 0),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 1, 0),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 0, 0),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 1, 0)),
           'American Put':(expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 0, 0),
                           fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 0, 0),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 1, 0),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 0, 0),
                            fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 1, 0))}
            
    fdm_ame_output = pd.DataFrame(data = fdm_ame)
    fdm_ame_output = fdm_ame_output[['FDM   ','Algorithm','American Call','American Put']] # rearranging columns
    print('\n 1(b). Finite Difference Methods (FDMs) for American Options')
    print('\n',fdm_ame_output,'\n')         

    # Question 2(a)
    
    print('------------------------------Question 2(a)----------------------------------')

    mc = {'European Call': monte_carlo_euler(S0,K,r,q,sigma,T,dt,1),
          'European Put': monte_carlo_euler(S0,K,r,q,sigma,T,dt,0),
          'Abs error_c':abs(monte_carlo_euler(S0,K,r,q,sigma,T,dt,1)-BSM_value(S0, K, r, q, sigma, T, 1)),
          'Abs error_p':abs(monte_carlo_euler(S0,K,r,q,sigma,T,dt,0)-BSM_value(S0, K, r, q, sigma, T, 0))}
    
    mc_output = pd.DataFrame(data = mc,index=[0])
    mc_output = mc_output[['European Call','Abs error_c','European Put','Abs error_p']]
    print('\n 2(a). Monte Carlo integration of risk neutral expectations (European Options)')
    print('\n',mc_output,'\n')
    
    # Question 2(b)
    
    print('\n------------------------------Question 2(b)----------------------------------')
    
    mc_reg = {'American Call': monte_carlo_reg2(S0,K,r,q,sigma,T,dt,1),
                       'American Put': monte_carlo_reg2(S0,K,r,q,sigma,T,dt,0)}
    mc_reg_output = pd.DataFrame(data = mc_reg,index=[0])

    print('\n 2(b). Monte Carlo Regression method II (American Options)')
    print('\n',mc_reg_output,'\n')
    
    # Question 3

    print('\n------------------------------Question 3------------------------------------')
    
    bs = {'European Call': BSM_value(S0, K, r, q, sigma, T, 1),
          'European Put': BSM_value(S0, K, r, q, sigma, T, 0)}
    bs_output = pd.DataFrame(data = bs,index=[0])
    print('\n 3. Closed-form solution formulas (European Options)')
    print('\n',bs_output,'\n')
    
    
    















