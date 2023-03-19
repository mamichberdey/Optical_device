import opticaldevicelib as od
import numpy as np
import os
import pandas as pd


"""Способ инициализации точечного источника c энергией En и распр. ВФ на расстоянии z от него"""
p = od.Point_source(z=100, En=10) 
E_arr = p.E()


N1_global = 100
Copy_flag = True
arr_len = 2 if Copy_flag else 2*N1_global

phases = 2*np.pi*np.random.rand(arr_len)
w_s = (np.random.rand(arr_len)-0.5)*np.pi/180*0 # повороты и сдвиги по нулям
s_s  = (np.random.rand(arr_len)-0.5)*2e-6*0

crl = od.CRL(lam=p.lam, arr_start=E_arr,\
                R=6.25e-6, A=50e-6, d=2e-6, N1=N1_global, z=0,\
                    molecula="Si", density=2.33, Flen=0, gap=0)

focus = crl.focus() 
crl.set_z(z=focus)

b_arr = np.linspace(1e-8, 0.4e-6, 2)
len_b = len(b_arr)
m_arr = np.linspace(1e6, 1e9, 2)

df = pd.DataFrame()
df["j"] = b_arr

df2 = pd.DataFrame()
df2["j"] = b_arr


for i, m in enumerate(m_arr):
    peak_slice = np.empty(len_b)
    for j, b in enumerate(b_arr):
        crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
                            R=6.25e-6, A=50e-6, d=2e-6, N1=N1_global, z=0,\
                                molecula="Si", density=2.33, Flen=0, gap=0,\
                                    b=b, m=m, copy=Copy_flag, arr_phase=phases, arr_s=s_s, arr_w=w_s)
        crlm.set_z(z=focus) 
        FWHM_crlm, peak_crlm = crlm.FWHM_max()
        peak_slice[j] = peak_crlm
    df[m] = peak_slice
    df2[m] = FWHM_crlm
    df.to_csv(os.path.abspath("output/3d_peak_output.csv"), index=False, sep=";")
    df2.to_csv(os.path.abspath("output/3d_FWHM_output.csv"), index=False, sep=";")
