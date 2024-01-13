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
                      or put option prices based on the dampening factor(alpha) give. For a positive alpha, it gives us 
                      the European call option price where for negative aplha, we will get the European put option 
                      price.The input parameters required are provided by the user.
                  
         
Numerical Methods   : Pricing Integrals using Fourier Transform and Inverse Fourier Transform are used. Trapezoid rule
                      is used for the summation.Conjugate properties are also used.
                      
Author              : Mohammed Ameer Khan

Date of Completion  : 22 November, 2018

Files Included      : main file, Python program file, Console output screenshot file and analysis document.
"""

"""
Importing libraries and functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fourier_transform_fft import *

"""
main function to call the function
"""

if __name__ == '__main__':
    
    S0 = 100
    K  = 80
    T  = 1
    r  = 0.05
    Kmin = 20
    B  = 50
    N  = 2**10
    sigma = 0.5
    alpha = np.array([2.5,5.0,10.0]) # an array of alpha values
    
    call_price = np.zeros(len(alpha))
    put_price = np.zeros(len(alpha))
    
    for i in range(len(alpha)):
        
        """
        Defining an array for call& put options and their strike prices
        """
        
        call_k = np.zeros(N)
        put_k = np.zeros(N)
        strike_call = np.zeros(N)
        strike_put = np.zeros(N)
        
        call_k, strike_call, graph_scale_call = fourier( alpha[i], S0, K, r, T, sigma, Kmin, B, N)
        put_k, strike_put, graph_scale_put = fourier( -alpha[i], S0, K, r, T, sigma, Kmin, B, N)
        
        
        """
        Approximating the strike price K for the option prices
        """
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        call_price[i] = approx(K, call_k, strike_call, 1)
        put_price[i] = approx(K, put_k, strike_put, 1)
        
        
        graph_fig = plt.figure()
        plt.suptitle ("European option prices for varying alphas = {0}".format(alpha[i], fontsize = 12))
        
        """
        Plotting the call option prices
        """

        call_graph = graph_fig.add_subplot(211)
        call_graph.plot(np.exp(call_k),strike_call/graph_scale_call,color="blue")
        call_graph.tick_params(labelsize=9)
        call_graph.set_xlabel("Strike price(K)",fontsize=9)
        call_graph.set_ylabel("Option price after scaling",fontsize=9)
        call_graph.set_title("Call option price after scaling = {0}".format(graph_scale_call),fontsize=10)

        """
        Plotting the put option prices
        """
        put_graph = graph_fig.add_subplot(212)
        put_graph.plot(np.exp(put_k),strike_put/graph_scale_put,color="red")
        put_graph.tick_params(labelsize=9)
        put_graph.set_xlabel("Strike price(K)",fontsize=9)
        put_graph.set_ylabel("Option price after scaling",fontsize=9)
        put_graph.set_title("Put option price after scaling = {0}".format(graph_scale_put),fontsize=10)

        graph_fig.subplots_adjust(hspace=1)
    plt.show()
    
    call_output= pd.DataFrame([call_price],index=["Call Prices"],columns=alpha)
    put_output= pd.DataFrame([put_price],index=["Put Prices"],columns=-alpha)
    print('\n',call_output,'\n')
    print('\n',put_output)
