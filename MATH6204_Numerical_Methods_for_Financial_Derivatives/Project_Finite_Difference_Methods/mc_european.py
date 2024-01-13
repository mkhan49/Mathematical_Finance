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
Importing numpy, scipy, and pandas libraries

"""
import numpy as np
import scipy.stats as stats

"""
Setting the seed for the random number generation
"""
np.random.seed(seed=123)

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
       

def BSM_value(S0, K, r, q, sigma, T, OptionInd):
    ''' Calculates Black-Scholes-Merton European call & put option value.

    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        maturity date
    r : float
        constant, risk-free interest rate
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
        European call value or put value depending on the Option Indicator
    '''
    
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * (T))/sigma * np.sqrt(T)
    
    d2 = d1 - sigma * np.sqrt(T)
    
    if OptionInd != 0 and OptionInd != 1:
        
        print("OptionInd has to be value 1 for Call option or 0 for Put option")
    
    elif OptionInd == 1: 
    
         option_value = S0 * np.exp (-q * (T)) * stats.norm.cdf(d1) - K * np.exp(-r * (T)) * stats.norm.cdf(d2)

    elif OptionInd == 0:
    
         option_value = -S0 * np.exp (-q * (T)) * stats.norm.cdf(-d1) + K * np.exp(-r * (T)) * stats.norm.cdf(-d2)
    
    return round(option_value,6)
                    
def monte_carlo_euler(S0,K,r,q,sigma,T,dt,OptionInd):
    ''' Calculates Monte Carlo simulated European call & put option value uding Euler discretization

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
        European call value or put value depending on the Option Indicator at time t
    '''
    
    NRepl = 500
    NSteps = int(T/dt)
    payoff = np.zeros((NRepl,1))
    paths = sample_paths_euler(S0,r,q,sigma,T,dt)

        
    if OptionInd != 0 and OptionInd != 1:
        
        print("OptionInd has to be value 1 for Call option or 0 for Put option")
    
    elif OptionInd == 1:
        
        for i in range(NRepl):
              
            payoff[i,0] = max(paths[i,NSteps]-K,0)
            
        option_value = (1/NRepl) * np.exp(-r*T) * np.sum(payoff)
        
        option_value_bsm = BSM_value(S0, K, r, q, sigma, T, 1)
        
        error = abs(option_value-option_value_bsm)
        

    elif OptionInd == 0:
        
        for i in range(NRepl):
              
            payoff[i,0] = max(K-paths[i,NSteps],0)
    
        option_value = (1/NRepl) * np.exp(-r*T) * np.sum(payoff)
        
        option_value_bsm = BSM_value(S0, K, r, q, sigma, T, 0)
        
        error = abs(option_value-option_value_bsm)
        
    #print ('%.3f      %.6f   %.6f   %.6f   Euler' % (dt,option_value, option_value_bsm, error))
          
    return round(option_value,6)

