import opticaldevicelib as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import time

time1 = time.perf_counter()

p = od.Point_source(z=100, En=10)
E_arr = p.E()

# # hole=Hole(lam=p.lam, arr_start=p.E(), D=5e-6, z=0)

# d=2*10**-6

crl = od.CRL(lam=p.lam, arr_start=E_arr,\
                R=6.25*10**-6, A=50*10**-6,\
                    d=2*10**-6, N1=10, z=0)


phases = 2*np.pi*np.random.rand(20)

crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
                    R=6.25*10**-6, A=50*10**-6,\
                        d=2*10**-6, N1=10, z=0,\
                            b=0.1*10**-6, m=0.3*10**6, copy=False, arr_phase=phases)  


# crlm_2 = od.CRLm(lam=p.lam, arr_start=E_arr,\
#                     R=6.25*10**-6, A=50*10**-6,\
#                         d=2*10**-6, N1=10, z=0,\
#                             b=0.1*10**-6, m=1*10**4, copy=False, arr_phase=phases) 




# # # crl3=CRL3(lam=p.lam, arr_start=p.E(),\
# # #       R=6.25*10**-6, A=50*10**-6,\
# # #       d=2*10**-6, N1=10, z=0)


focus = crl.focus() 
print(focus)
crl.set_z(z=focus) 
crlm.set_z(z=focus)
# crlm_2.set_z(z=focus)


int_ideal = crl.I()
int_mid = crlm.I()
# int_low = crlm_2.I()

# arr = np.zeros(crl.N)

# for i in range(1):
#     arr += crlm.T(phase=2*np.pi*np.random.rand())-crl.T()
print("FWHM_ideal = ", crl.FWHM(p.x, int_ideal ), "FWHM_mid = ", crl.FWHM(p.x, int_mid))

# plt.plot(arr)

plt.plot(p.x, crl.analytic_solution_CRL(z0=p.z, z1=crl.z), label="analytical CRL")
# plt.yscale('log')
plt.plot(p.x, int_ideal, label="numerical CRL")
plt.plot(p.x, int_mid, label="numerical CRLm")
# plt.plot(p.x, int_low, label="numerical CRLm_2")
plt.legend()
plt.grid()
# plt.xlim([-4e-5, 4e-5])
# plt.ylim([10**-9, 10**3])
# plt.xlim([-0.25e-6, 0.25e-6])
# plt.ylim([-20, 250])
# print("mean = "+str(np.mean(arr[]))) 

time2 = time.perf_counter()
print("t = "+str(time2-time1))
plt.show()