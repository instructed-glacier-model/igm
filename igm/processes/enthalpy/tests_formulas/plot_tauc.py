#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt

def cela(s,P0):

    Pe0 = 0.69
    PCc = 0.12
    Pdelta = 0.02 
    PN0 = 1000 # Pa
    
    phi = 30 
    RE = np.tan(phi*math.pi/180)

    gt = PN0 * ( Pdelta * P0 / PN0 )**s * 10**( Pe0 * (1-s) /PCc )

    return np.minimum(P0, gt)*RE

def overpressure(thk):
    rho=917
    g = 9.81
    return thk * rho * g

RE = np.tan(30*math.pi/180)

thk=1000
P0 = overpressure(thk)

s=np.arange(0,101)/100

tauc = cela(s,P0)

plt.plot(s,tauc)
plt.savefig('tauc_fct_of_s.png')

print('b1 :',P0*0.02*RE,P0*RE)
print('min max :',np.min(tauc),np.max(tauc))
print('beg last :',tauc[0],tauc[-1])
