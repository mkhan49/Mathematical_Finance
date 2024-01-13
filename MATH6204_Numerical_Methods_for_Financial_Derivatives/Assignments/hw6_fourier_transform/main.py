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
                      a way that we can make use of fourier transform provied the characteristic function is known.
                      The pricing integrals using the Foruier transform and inverse Fourier transform are derived and
                      then the summation is approximated using the Trapezoidal rule. Half frequency domain and Full
                      frequency domain are used to compute the option prices. The algorithm gives us the European call
                      or put option prices based on the dampening factor(alpha) give. FOr a positive alpha, it gives us 
                      the European call option price where for negative aplha, we will get the European put option 
                      price.The input parameters required are provided by the user.
                  
         
Numerical Methods   : Pricing Integrals using Fourier Transform and Inverse Fourier Transform are used. Trapezoid rule
                      is used for the summation.Conjugate properties are also used.
                      
Author              : Mohammed Ameer Khan

Date of Completion  : 21 November, 2018

Files Included      : main file, Python program file, Console output screenshot file and analysis document.
"""

"""
Importing libraries and functions
"""
import numpy as np
import pandas as pd

from fourier_transform import *

"""
main function to call the function
"""

if __name__ == '__main__':
    
    S0 = 100
    K  = 80
    T  = 1
    r  = 0.05
    B  = 50
    N  = 1000
    sigma = 0.5
    alpha = np.array([2.5,5,10]) # an array of alpha values for call options


    # Creating a tablular format for call values

    call_values = {'alpha':np.array([2.5,5,10]),'Full Frequency Domain': fourier(alpha, S0, K, r, T, sigma, B, N, 1) , 
              'Half Frequency Domain':fourier(alpha, S0, K, r, T, sigma, B, N, 0)}

    call_output = pd.DataFrame(data = call_values,index=np.array([2.5,5,10]))

    call_output.set_index('alpha',inplace=True)

    print('\n Table 1: European Call option values using Fourier Transform for different frequency domains(alpha)')

    print('\n',call_output,'\n')
    
    alpha = np.array([-2.5,-5,-10]) # an array of alpha values for put options

    # Creating a tablular format for put values

    put_values = {'alpha':np.array([-2.5,-5,-10]),'Full Frequency Domain': fourier(alpha, S0, K, r, T, sigma, B, N, 1) , 
              'Half Frequency Domain':fourier(alpha, S0, K, r, T, sigma, B, N, 0)}

    put_output = pd.DataFrame(data = put_values,index=np.array([-2.5,-5,-10]))

    put_output.set_index('alpha',inplace=True)

    print('\n Table 2: European Put option values using Fourier Transform for different frequency domains(alpha)')

    print('\n',put_output)
