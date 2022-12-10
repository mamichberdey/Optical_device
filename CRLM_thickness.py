import opticaldevicelib as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os


time1 = time.perf_counter()

p = od.Point_source(z=100, En=10)
E_arr = p.E()


crl = od.CRL(lam=p.lam, arr_start=E_arr,\
            R=6.25*10**-6, A=50*10**-6,\
                d=2*10**-6, N1=10, z=0)

crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
                    R=6.25*10**-6, A=50*10**-6,\
                        d=2*10**-6, N1=10, z=0,\
                            b=0.1*10**-6, m=0.5*10**8, copy=False, arr_phase=2*np.pi*np.random.rand(20)) 


crlm_2 = od.CRLm(lam=p.lam, arr_start=E_arr,\
                    R=6.25*10**-6, A=50*10**-6,\
                        d=2*10**-6, N1=10, z=0,\
                            b=0.4*10**-6, m=0.5*10**8, copy=False, arr_phase=2*np.pi*np.random.rand(20)) 



Ti=crl.T()

phase_1 = 2*np.pi*np.random.rand()
# phase_1 = 2*np.pi*0.92
phase_2 = 2*np.pi*np.random.rand()
# phase_2 = phase_1 + 2*np.pi*1e-4


# phase_2 = 0.25*np.pi

print(phase_1/(2*np.pi), phase_2/(2*np.pi))



Ts = crlm.T(phase=phase_1)+crlm.T(phase=phase_2)
Ts_2 = crlm_2.T(phase=phase_1)+crlm_2.T(phase=phase_2)
# Ts = crlm.T(x=p.x, R=crlm.R, A=crlm.A, d=crlm.d, phase=2*np.pi*np.random.rand())
mas = 10**6
plt.plot(p.x*mas, Ts*mas, "k", linewidth=0.9)
plt.plot(p.x*mas, Ts_2*mas, "k", linewidth=0.9)
# plt.plot(p.x*mas, Ti*mas, "r", linewidth=0.9)
plt.ylim([-2, 55-2])
plt.xlim([-55, 55])
plt.grid()
plt.xlabel(r"x, мкм")
plt.ylabel(r"t(x), мкм")
# plt.legend()
# df = pd.DataFrame({"x, мкм": p.x*mas, "t(x), мкм": Ts*mas})
# df.to_csv(os.path.abspath("output/T.csv"), index=False, sep=";")
# plt.title("One element of an X-ray compound refractive lens half thickness")
time2 = time.perf_counter()
print("t = "+str(time2-time1))   
# plt.savefig(os.path.abspath("output/T.png"), format="png", dpi=800)
plt.show()
