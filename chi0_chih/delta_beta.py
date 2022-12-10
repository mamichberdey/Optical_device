import numpy as np


# Silicon

ro = 2.33
M = 28.0855
e = 20.087

f1 = 14.047600
f2 = 0.052853800

co = [-0.414970011, 1.34870005, -0.222310007, 8.419600315E-03]

df2 = (1.4312e-5*e*np.exp(co[0]+co[1]*np.log(e) +
       co[2]*np.log(e)**2+co[3]*np.log(e)**3))
chi0 = -8.3036e-4 * ro/M/e**2 * (f1-1j*(f2+df2))
delta = abs(chi0.real/2)
beta = abs(chi0.imag/2)
print(delta, beta)
