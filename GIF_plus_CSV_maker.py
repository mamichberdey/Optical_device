import opticaldevicelib as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os

time1 = time.perf_counter()

p = od.Point_source(z=100, En=20) # источник света
E_arr = p.E() # распр вф света на начале линзы

crl = od.CRL(lam=p.lam, arr_start=E_arr,\
                R=6.25*10**-6, A=50*10**-6,\
                    d=2*10**-6, N1=100, z=0)

focus = crl.focus() # смена расстояния с нуля до фокуса линзы
crl.set_z(focus) 

crl_I = crl.I()

df = pd.DataFrame({"x": p.x, "y_crl": crl_I}) # начало создания df и gif-картинки
df_phases = pd.DataFrame()
fig, ax = plt.subplots()
def animate(ap):
    ax.clear()

    # crl=CRL(lam=p.lam, arr_start=p.E(),\
    #             R=6.25*10**-6, A=50*10**-6,\
    #                 d=2*10**-6, N1=10, z=focus)
    
    phases = 2*np.pi*np.random.rand(200)
    
    df_phases["i = "+str(ap)] = phases

    crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
                        R=6.25*10**-6, A=50*10**-6,\
                            d=2*10**-6, N1=100, z=focus,\
                                b=0.1*10**-6, m=0.25*10**6, copy=False, arr_phase=phases)  

    I = crlm.I()

    ax.plot(p.x, crl_I, color="g", label="numerical solution CRL", alpha=0.9, linewidth=0.8) 

    line = ax.plot(p.x, I, color="r", label="numerical solution CRLm", alpha=0.9, linewidth=0.8)
    
    ax.set_title("i = "+format(ap, '.2f'))
    df["ph2 = "+str(ap)] = I
    ax.set_xlabel(r"x, m")
    ax.set_ylabel(r"Intensity")
    ax.legend()
    ax.set_xlim([-0.5e-6, 0.5e-6])
    ax.set_ylim([-20, 250])
    ax.grid()
    return line

sin_animation = animation.FuncAnimation(fig, animate, 
                                            frames=np.linspace(0, 2*np.pi, 1),
                                                interval=30, repeat=False)

sin_animation.save(os.path.abspath("output/anim.gif"), writer='pillow', fps=30, dpi=400) 
df.to_csv(os.path.abspath("output/output.csv"), index=False, sep=";")
df_phases.to_csv(os.path.abspath("output/output_phases.csv"), index=False, sep=";")
time3 = time.perf_counter()

print("t = "+str(time3-time1))  # конец создания df и gif-картинки