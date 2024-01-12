#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                             MATH 6205 - Numerial Methods for Financial Derivatives
                                                  Fall 2018


Purpose             : The objective of this Python program is to calculate the price of European Call and Put options 
                      by making use of the closed form solution provided by the classical Black-Scholes equation with
                      the input parameters provided by the user.
         
         
Numerical Methods   : The classical Black-Scholes equation for option pricing is used. 
    
    
Author              : Mohammed Ameer Khan

Date of Completion  : 10 September, 2018

Files Included      : Python program file and Console output screenshot file
"""

# Importing numpy, scipy, and pandas libraries

import numpy as np
import scipy.stats as stats
import pandas as pd

# Defining a function df to calculate the value of d1
 
def df(St, K, t, T, r, q, sigma):
    
    d1 = (np.log(St/K) + (r - q + 0.5 * sigma**2) * (T-t))/sigma * np.sqrt(T-t)
    
    return d1

# Defining a function std_norm_cdf to approximate standard normal cumulative distribution
    
def std_norm_cdf(x):
    
    # Defining parameter values 
    
    z   =  1 / (1 + 0.2316419 * abs(x))
    a1  =  0.319381530
    a2  = -0.356563782
    a3  =  1.781477937
    a4  = -1.821255978
    a5  =  1.330274419
  
    f_x = 1/ np.sqrt(2 * np.pi) * np.exp(- (x**2) / 2) 
    
    s = ((((a5 * z + a4) * z + a3) * z + a2) * z + a1)
    
    if x >= 0 :
        cdf = 1 - f_x * z * s
    else :
        cdf = f_x * z * s
        
    return cdf

# Defining a function to calculate the Call & Put prices using Black-Scholes equation

def BSM_value(St, K, t, T, r, q, sigma, OptionInd):
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
        European call value or put value depending on the Option Indicator at time t
    '''
    
    d1 = df(St, K, t, T, r, q, sigma)
    
    d2 = d1 - sigma * np.sqrt(T-t)
    
    if OptionInd != 0 and OptionInd != 1:
        
        raise ValueError("OptionInd has to be value 1 for Call option or 0 for Put option")
    
    elif OptionInd == 1: 
        
    # stats.norm.cdf gives the cumulative standard normal distribution value
    
         option_value = St * np.exp (-q * (T-t)) * stats.norm.cdf(d1) - K * np.exp(-r * (T-t)) * stats.norm.cdf(d2)

    elif OptionInd == 0:
    
         option_value = -St * np.exp (-q * (T-t)) * stats.norm.cdf(-d1) + K * np.exp(-r * (T-t)) * stats.norm.cdf(-d2)
    
    return option_value
 
"""    
 Output generation
 
   """ 
# Table 1: Verification of cumulative standard normal distribution values using scipy and the approximation function


x = np.vectorize(std_norm_cdf) # Vectorizing to compare values

d = np.array([-2,2,1]) # Set of input values of x to calculate cdf

x(d)
   
std_norm = {' N(x)-Scipy':stats.norm.cdf(d),' N(x)-Approx fun':x(d),'Difference':stats.norm.cdf(d)-x(d),
            'x value': np.array([-2,2,1])}


cdf = pd.DataFrame(data=std_norm,index = np.array([-2,2,1]) )

cdf.set_index('x value',inplace=True)

print('\n Table 1: Comparing N(x) values using Approximation function and Scipy')

print('\n',cdf)

# Table 2: Initializing parameters

St = 100
K = 100
T = 1
t = 0
r = 0.05
q = 0.025
sigma = np.arange(0.1,1.1,0.1) # an array of volatility values ranging from 0.1 to 1

# Creating a tablular format for call and put values

values = {'Volatility': np.arange(0.1,1.1,0.1),'Call values': BSM_value(St, K, t, T, r, q, sigma,1) , 
     'Put Values': BSM_value(St, K, t, T, r, q, sigma,0)}

output = pd.DataFrame(data = values,index=np.arange(0.1,1.1,0.1))

output.set_index('Volatility',inplace=True)

print('\n Table 2: European Call & Put option values at time t for different volatilities')

print('\n',output)