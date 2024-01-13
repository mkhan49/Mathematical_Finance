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


Purpose             : The objective of this Python program is to compute the prices of European calls and puts using  
                      Fourier transform techniques. Geometric Brownian Motion as the underlying diffusion process 
                      for the Stock price. Under risk neutral evaluation, the Brownian Motion is transformed in such 
                      a way that we can make use of Fourier transform provided the characteristic function is known.
                      The pricing integrals using the Fourier transform and inverse Fourier transform are derived and
                      then the summation is approximated using the Trapezoidal rule. Half frequency domain and Full
                      frequency domain are used to compute the option prices. The algorithm gives us the European call
                      or put option prices based on the dampening factor(alpha) give. For a positive alpha, it gives us 
                      the European call option price where for negative alpha, we will get the European put option 
                      price.The input parameters required are provided by the user.
                  
         
Numerical Methods   : Pricing Integrals using Fourier Transform and Inverse Fourier Transform are used. Trapezoid rule
                      is used for the summation.Conjugate properties are also used.
                      
Author              : Mohammed Ameer Khan

Date of Completion  : 21 November, 2018

Files Included      : main file, Python program file, Console output screenshot file and analysis document.
"""

"""
Importing numpy, scipy, and pandas libraries

"""
import numpy as np


"""
Function to calculate the characteristic function 
"""
def characteristic_fn(alpha, S0, K, r, T, w, sigma):
    '''

    Parameters
    ==========
    alpha : float
          damping factor
    S0 : float
        stock price at time 0
    K : float
        exercise price
    T : float
        maturity date
    r : float
        constant, risk free rate
    w : float
        frequency
    sigma : float
            volatility
            
    ''' 
    
    w_new =  w + (alpha+1)*1j
    
    fn = np.exp(-r*T) * np.exp((-1j*(np.log(S0)+(r - sigma**2/2)*T)*w_new)-sigma**2/2 *T * w_new**2)
    
    return fn

"""
Function to calculate the denominator of the v function
"""

def denominator(alpha,w):
    
    d = (alpha-1j*w) * (alpha-1j*w+1)
    
    return d  

"""
Function to calculate the fourier transformation based on the Trapezoid algorithm
"""

def fourier(alpha, S0, K, r, T, sigma, B, N, FreqInd):
    '''

    Parameters
    ==========
    alpha : float
          damping factor
    S0 : float
        stock price at time 0
    K : float
        exercise price
    T : float
        maturity date
    r : float
        constant, risk free rate
    B : float
        limit of bandwidth
    N : integer
        # of frequencies
    sigma : float
            volatility

    FreqInd : integer
                1 - corresponds to Full Frequency Domian
                0 - corresponds to Half Frequency Domian
                
    Returns
    =======
    option_value : float
        European call value or put value depending on the Frequency Indicator
        
    '''
    
    h = B/N
    k = np.log(K)
    v_half = 0
    v_full = 0
    x = np.exp(-alpha*k)/np.pi
       
    if FreqInd != 0 and FreqInd != 1:
        
        print("FreqInd has to be value 1 for Full Frequency Domain or 0 for Half Frequency Domain")
    
    elif FreqInd == 0:
        
        for m in range(N+1):
            
            w = h*m
            
            if m==0  or m==N:
                dw = h/2
            else:
                dw = h
                
            v_half += np.exp(1j*w*k) * dw * characteristic_fn(alpha, S0, K, r, T, w, sigma) / denominator(alpha,w)
            
        option_value = np.real(x * v_half)
            
    elif FreqInd == 1:
        
        for m in range(-N,N+1):
            
            w = h*m
            
            if m==-N  or m==N:
                dw = h/2
            else:
                dw = h
                
            v_full += np.exp(1j*w*k) * dw* characteristic_fn(alpha, S0, K, r, T, w, sigma) / denominator(alpha,w)
            
        option_value = np.real(x/2 * v_full)
    
    return option_value

    
    
    
    
