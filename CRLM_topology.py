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
                            b=0.4*10**-6, m=0.5*10**8, copy=False, arr_phase=2*np.pi*np.random.rand(20)) 


X1, Y1 = crlm.wavy_parabola(phase=2*np.pi*np.random.rand())
X2, Y2 = crlm.wavy_parabola(phase=2*np.pi*np.random.rand())

print(len(X2), len(Y2))
mas = 10**6
plt.plot((Y1)*mas, X1*mas, color="k", linewidth=0.9)
plt.plot(-(Y2)*mas, X2*mas, color="k", linewidth=0.9)
plt.plot(crl.T()*mas, p.x*mas, "r", linewidth=0.9)
plt.plot(-crl.T()*mas, p.x*mas, "r", linewidth=0.9)
plt.ylim([-55, 55])
plt.xlim([-55, 55])
plt.xlabel(r"z, мкм")
plt.ylabel(r"x, мкм")
plt.grid()
# plt.title("One element of an X-ray compound refractive lens (CRL)")  
df = pd.DataFrame({"x, µm": X1*mas, "y, µm": Y1*mas})
df.to_csv(os.path.abspath("output/CRL.csv"), index=False, sep=";")
plt.savefig(os.path.abspath("output/CRL.png"), format="png", dpi=800)
plt.show()