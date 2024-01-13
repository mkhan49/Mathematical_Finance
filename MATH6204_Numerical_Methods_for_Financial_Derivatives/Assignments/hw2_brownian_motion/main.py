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
Importing sample_path function 

"""
from brownian_motion import *

"""
main function to call the function
"""

if __name__ == '__main__':
    sample_path(100, 0.1, 0.025, 0.2, 1,)  
           
