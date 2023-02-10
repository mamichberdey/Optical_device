import opticaldevicelib as od
import numpy as np
import matplotlib.pyplot as plt
from sympy import solve
import time


time1 = time.perf_counter()

p = od.Point_source(z=100, En=10)
E_arr = p.E()

# # hole=Hole(lam=p.lam, arr_start=p.E(), D=5e-6, z=0)

# d=2*10**-6

N1_global = 10

crl = od.CRL(lam=p.lam, arr_start=E_arr,\
                R=5e-6, A=24e-6, d=5e-6, N1=N1_global, z=0,\
                     molecula="C21SO8SiH36", density=1.12, Flen=0, gap=0)




# a = np.random.rand(10)
# w_s[1::2] = -w_s[0::2]

# print(a)


focus = crl.focus()
print(focus)
crl.set_z(z=focus) 

D = 2e-6
R=5e-6
A=24e-6
d=5e-6
P = A**2/(4*R)+d

L = N1_global*P

phases = 2*np.pi*np.random.rand(2*N1_global)*0
# w_s = (np.random.rand(2*N1_global)-0.5)*np.pi/180*15
# w_s = np.linspace(0, np.pi/180*5, 2*N1_global)
w_s = np.linspace(-4*D/L, 4*D/L, N1_global*2)
# w_s = np.pi/180*np.ones(2)*1/10

print(w_s)
# s_s = (np.random.rand(2*N1_global)-0.5)*2e-6*0
s_s = L**2/(8*D)*(np.cos(w_s)-1)-D/2

w_s[1::2] = w_s[0::2]

w_s = w_s*0

# plt.plot(s_s)
# plt.show()

crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
                    R=5*10**-6, A=24*10**-6,\
                        d=5*10**-6, N1=N1_global, z=0,\
                            b=1*10**-6*0, m=3*10**6, copy=False, arr_phase=phases, arr_s=s_s, arr_w=w_s,\
                                  molecula="C21SO8SiH36", density=1.12, Flen=0, gap=0)

crlm.set_z(z=focus)



# angle = np.pi/180*15
# # angle = 0
# plt.plot(p.x, crlm.T(phase=0, s=1e-5, w=angle))
# plt.plot(p.x, crlm.T(phase=0, s=-1e-5, w=angle))
# # plt.scatter(crlm.x_arr(p.x, phase=0, s=0, w=angle), crlm.y_arr(p.x, phase=0, w=angle) , s=1, color="g")
# plt.show()

plt.plot(p.x, crl.I(), label="numerical CRL")
plt.plot(p.x, crlm.I(), label="numerical CRLm")
plt.legend()
plt.grid()
time2 = time.perf_counter()

print("time = ", time2-time1)

plt.show()
