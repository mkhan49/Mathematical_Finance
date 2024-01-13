#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 02:57:10 2018

@author: khan
"""

import numpy as np

def brennan_solver(A, b, g):
    '''
    Solve the linear system Ax = b using the Brennan algorithm.
    This is an efficient solver for tridiagonal A matrices.
    It solves in two steps. A forward step that reduces the system
    in a way that all subdiagonal terms become 0. And a backwards
    step that solves the reduced system from the bottom up.
    Parameters
    ----------
    A : Numpy matrix
        A tridiagonal matrix to solve using the Thomas algorithm
    b : Numpy array
        The right hand side of Ax = b
    g : Numpy 1D array
        The vector to elementwise take the max against at each iteration
    Returns
    -------
    x : Numpy array
        The solution to the linear system.
    '''

    [A_reduced, b_reduced] = forward_step(A.copy(), b.copy())

    x = backward_step(A_reduced, b_reduced, g.copy())

    return x

def forward_step(A, b):

    N = b.shape[0]

    for i in range(1, N): # 1 to N-1

        alpha_i   = A[i,     i    ]
        alpha_i_1 = A[i - 1, i - 1]
        beta_i_1  = A[i - 1, i    ]
        gamma_i   = A[i,     i - 1]

        # Alter alpha
        A[i, i] = alpha_i - beta_i_1 * (gamma_i / alpha_i_1)

        # Set gamma to 0
        A[i, i-1] = 0

        # Alter b
        b[i] = b[i] - b[i-1] * (gamma_i / alpha_i_1)

    return A, b

def backward_step(A_reduced, b_reduced, g):

    N = b_reduced.shape[0]
    x = np.zeros(N)

    # Set the last value of x, known
    x[N-1] = np.maximum(g[N-1], b_reduced[N-1] / A_reduced[N-1, N-1])

    for i in reversed(range(N-1)): # N-2 to 0

        b_i     = b_reduced[i]
        beta_i  = A_reduced[i, i + 1]
        alpha_i = A_reduced[i, i]

        # Iterate x
        x[i] = np.maximum(g[i], (b_i - beta_i * x[i + 1]) / alpha_i)

    return x