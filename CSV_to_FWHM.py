import opticaldevicelib as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os

time1 = time.perf_counter()


df = pd.read_csv("output/output.csv", delimiter=";")

p = od.Point_source(z=100, En=20)
E_arr = p.E()


crl = od.CRL(lam=p.lam, arr_start=E_arr,\
                R=6.25*10**-6, A=50*10**-6, d=2*10**-6, N1=100, z=0)

focus = crl.focus() # смена расстояния с нуля до фокуса линзы
crl.set_z(focus) 

crlm2c_zero = od.CRLm(lam=p.lam, arr_start=E_arr,\
                    R=6.25*10**-6, A=50*10**-6,\
                        d=2*10**-6, N1=100, z=focus,\
                            b=0.1*10**-6, m=0.25*10**6, copy=True, arr_phase=np.zeros(2)) 

I_low = crlm2c_zero.I()

crlm2c_pi = od.CRLm(lam=p.lam, arr_start=E_arr,\
                    R=6.25*10**-6, A=50*10**-6,\
                        d=2*10**-6, N1=100, z=focus,\
                            b=0.1*10**-6, m=0.25*10**6, copy=True, arr_phase=np.array([0, np.pi])) 

I_peak = crlm2c_pi.I()







table = df.to_numpy()

xx = table[:, 0]


plt.plot(xx*10**6, crl.analytic_solution_CRL(z0=p.z, z1=crl.z), label="analytical CRL", alpha=0.9, linewidth=0.8) 
plt.plot(xx*10**6, table[:, 1], label="numerical solution CRL", alpha=0.9, linewidth=0.8)
plt.plot(xx*10**6, I_low, label="low", alpha=0.9, linewidth=0.8)
plt.plot(xx*10**6, I_peak, label="peak", alpha=0.9, linewidth=0.8)




for i in range(2, len(table[0, :])):
    plt.plot(xx*10**6, table[:, i], color="k", alpha=0.9, linewidth=0.8)
    
    
    
    
plt.xlabel(r"x, мкм")
plt.ylabel(r"I, отн. ед.")
plt.legend()
plt.grid()
plt.xlim([-0.1, 0.1])
plt.ylim([-2, 240])
plt.savefig(os.path.abspath("output/random_200.png"), format="png", dpi=800)
plt.show()
