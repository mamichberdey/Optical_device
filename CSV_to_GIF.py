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


focus = crl.focus() # смена расстояния с нуля до фокуса линзы
crl.set_z(focus) 

table = df.to_numpy()

xx = table[:, 0]
fwhm = []
peaks = []

# b_arr = np.linspace(1e-8, 0.4e-6, 25)
# m_arr = np.linspace(1e6, 1e9, 25)
# ph_arr = np.linspace(0, 2*np.pi, 300)
# m_arr = m_arr[2:]


# print(m_arr)

# fwhm_crl = crl.FWHM(y_arr=table[:, 1], x_arr=xx)
# peak_crl = np.max(table[:, 1])

# for i in range(2, len(table[0, :])):
#     gauss = table[:, i]
#     peaks.append(np.max(gauss))
#     fwhm.append(crl.FWHM(y_arr=gauss, x_arr=xx))

fig, ax = plt.subplots()
def animate(ap):
    ax.clear()
    ax.plot(xx*10**6, crl.analytic_solution_CRL(z0=p.z, z1=crl.z), label="analytical CRL", alpha=0.9, linewidth=0.8) 
    ax.plot(xx*10**6, table[:, 1], label="numerical solution CRL", alpha=0.9, linewidth=0.8)
    ax.plot(xx*10**6, I_low, label="low", alpha=0.9, linewidth=0.8)
    ax.plot(xx*10**6, I_peak, label="peak", alpha=0.9, linewidth=0.8)
    line = ax.plot(xx*10**6, table[:, ap], label="numerical solution CRLm", alpha=0.9, linewidth=0.8)
    # ax.set_yscale('log')
    # ax.set_title(r"ν = "+format(m_arr[ap-2]*10**-6, '4.2f')+r" ${мкм}^{-1}$"+"\n"+"A = 0.1 мкм")
    # ax.set_title(r"A = "+format(b_arr[ap-2]*10**6, '4.2f')+r" ${мкм}^{}$"+"\n"+"ν = 10"+r" ${мкм}^{-1}$")    
    # ax.set_title(r"φ = "+format(ph_arr[ap-2]/np.pi, '4.2f')+"π,"+"\n"+"A = 0.1"+r" ${мкм}^{},$"+" ν = 10"+r" ${мкм}^{-1}$")
    # ax.set_title("частота = "+str(ap)+r" ${мкм}^{-1}$")
    ax.set_xlabel(r"x, мкм")
    ax.set_ylabel(r"I, отн. ед.")
    ax.legend()
    ax.grid()
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-2, 240])
    # ax.set_ylim([10**-9, 10**3])

    return line

sin_animation = animation.FuncAnimation(fig, animate, 
                                            frames=np.arange(2, len(table[0, :])),
                                                interval=30, repeat=True)

sin_animation.save(os.path.abspath("output/anim_changed.gif"), writer='pillow', fps=10, dpi=400) 

time3 = time.perf_counter()
print(f't = {time3-time1}')  # конец создания df и gif-картинки

