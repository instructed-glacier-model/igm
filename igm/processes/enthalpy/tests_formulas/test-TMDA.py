#!/usr/bin/env python3

import tensorflow as tf

from igm.processes.process.enthalpy import *

M = tf.ones((10))
U = -2 * tf.ones((9))
L = -2 * tf.ones((9))
R = tf.Variable([4, 6, 7, 8, 9, 4, 6, 7, 8, 9.03])

RES = solve_TDMA(L, M, U, R)

A = tf.Variable(tf.zeros((10, 10)))

for i in range(10):
    A[i, i].assign(M[i])

for i in range(9):
    A[i, i + 1].assign(U[i])
    A[i + 1, i].assign(L[i])

tf.multiply(A, RES)

RHS = tf.linalg.matvec(A, RES)

print(RHS)
