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
                      is applied at each stopping time only for the in the money options so that it will reduce the 
                      computation time as well. Theindependent variable (x) will be our stock values generated using 
                      the Euler discretization and the dependent variable (y) will be the discounted payoff values from 
                      the immediate next time step.This process is applied to all the time steps and then the average 
                      of initial time step (t0) gives us the price of the American option at time, t0. We have used a 
                      third order polynomial for the regression and the Least Squares method is used to find the optimal 
                      coefficients.
         
         
Numerical Methods   : Geometric Brownian Motion is used for the simulation of sample paths. Euler discretization is used 
                      as a discretization method to discretize the SDE of the stock process. Monte Carlo simulations is 
                      used for the valuation of American options. Regression is used at each stopping times and Least Squares
                      is to optimize the regression coefficients.
    
    
Author              : Mohammed Ameer Khan

Date of Completion  : 28 November, 2018

Files Included      : main file, Python program file, Console output screenshot file and analysis document.
"""

"""
Importing numpy, scipy, and pandas libraries

"""
import numpy as np
import scipy as scipy

"""
Setting the seed for the random number generation
"""
np.random.seed(seed=123)

"""
Function to calcualte the sample paths using Euler discretization

"""

def sample_paths_euler(S0,r,q,sigma,T,dt):
    ''' Simulates the path for Stock price using Euler discretization.

    Parameters
    ==========
    S0 : float
        stock price at time 0
    T : float
        maturity date
    dt : float
         time step
    r : float
        constant, risk free rate
    q : float
        constant, time-continuous dividend yield
    sigma : float
            volatility

    ''' 
    np.random.seed(seed=123)
    NRepl = 500
    NSteps = int(T/dt)
    St = np.zeros((NRepl,NSteps+1))
    St[:,0] = S0
    for i in range(NRepl):
        for j in range(NSteps):
            St[i,j+1] = St[i,j] + (r-q) * St[i,j] * dt + sigma * St[i,j] * dt**0.5 * np.random.randn()
    
    return St

"""
Function to generate the third order polynomial function
"""

def model(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2 + a3*x**3

"""
Function to calculate the Monte Carlo regerssion method I values for American options
"""

def monte_carlo_reg2(S0,K,r,q,sigma,T,dt,OptionInd):
    ''' Calculates Monte Carlo simulated  American call & put option value using Euler discertization

    Parameters
    ==========
    S0 : float
        stock price at time 0
    K : float
        exercise price
    T : float
        maturity date
    dt : float
         time step
    r : float
        constant, risk free rate
    q : float
        constant, time-continuous dividend yield
    sigma : float
            volatility

    OptionInd : integer
                1 - corresponds to Call Value
                0 - corresponds to Put value
                
    Returns
    =======
    option_value : float
        American call value or put value depending on the Option Indicator at time t
    '''
    NRepl = 500
    NSteps = int(T/dt)
    payoff_call = np.zeros(NRepl)
    payoff_put = np.zeros(NRepl)
    paths = sample_paths_euler(S0,r,q,sigma,T,dt)
    time_call = np.zeros(NRepl)
    time_put = np.zeros(NRepl)
    price_call = 0
    price_put = 0
    
     
    if OptionInd != 0 and OptionInd != 1:
        
        print("OptionInd has to be value 1 for Call option or 0 for Put option")
    
    elif OptionInd == 1:
            
        payoff_call[:] =  np.maximum(paths[:,NSteps]-K,0)
        time_call[:] = NSteps
            
        for j in range(NSteps-1,0,-1):
            
            data = np.zeros((NRepl,2))
            
            for k in range(NRepl):
                
                if paths[k,j]>K: #data only for the in the money options
                    data[k,0] = paths[k,j] 
                    data[k,1] = np.exp(-r*dt*(time_call[k]-j)) * payoff_call[k]
            
            data_final = data[data[:,0]>0]
            xdata = data_final[:,0]
            ydata = data_final[:,1]
                    
            popt,pcov = scipy.optimize.curve_fit(model,xdata,ydata) 
            
            for l in range(NRepl):
                if paths[l,j]>K:    
                    val = model(paths[l,j],popt[0],popt[1],popt[2],popt[3])
                    if max(paths[l,j]-K,0) >= val:
                        payoff_call[l] = max(paths[l,j]-K,0) #updating payoff function
                        time_call[l] = j # updating timestep

        for m in range(NRepl):
            
            price_call += np.exp(-r*time_call[m]*dt)  * payoff_call[m]
            
        option_value = (1/NRepl) * price_call

    
    elif OptionInd == 0:
            
        payoff_put[:] =  np.maximum(K-paths[:,NSteps],0)
        time_put[:] = NSteps
            
        for j in range(NSteps-1,0,-1):
            
            data = np.zeros((NRepl,2))
            
            for k in range(NRepl):
                
                if (K-paths[k,j])>0: #data only for the in the money options
                    data[k,0] = paths[k,j]
                    data[k,1] = np.exp(-r*dt*(time_put[k]-j)) * payoff_put[k]
            
            data_final = data[data[:,0]>0]
            xdata = data_final[:,0]
            ydata = data_final[:,1]
                    
            popt,pcov = scipy.optimize.curve_fit(model,xdata,ydata) 
            
            for l in range(NRepl):
                if K>paths[l,j]:    
                    val = model(paths[l,j],popt[0],popt[1],popt[2],popt[3])
                    if max(K-paths[l,j],0) >= val:
                        payoff_put[l] = max(K-paths[l,j],0) #updating payoff function
                        time_put[l] = j # updating timestep

        for m in range(NRepl):
            
            price_put += np.exp(-r*time_put[m]*dt)  * payoff_put[m]
            
        option_value = (1/NRepl) * price_put

    
    return round(option_value,6)
