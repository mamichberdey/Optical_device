---
jupyter:
  kernelspec:
    display_name: Python 3.10.0 64-bit
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.0
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: 27f6fea6f47ae512550f0b8facdbd035a93e1dd89633f7bf2dd00a2502c71d0d
---

::: {.cell .markdown}
Python library for modeling and analyzing optical circuits with optical
device. The library is based on OOP, optical device lib allows you to
build a circuit with CRL lenses for X-ray radiation.
:::

::: {.cell .markdown}
Тестовая реализация кода с помощью opticaldevicelib
:::

::: {.cell .code execution_count="1"}
``` python
import opticaldevicelib as od
import numpy as np
import matplotlib.pyplot as plt
od.Optical_device.set_value(new_dx=5e-9, new_N=2**15)
```
:::

::: {.cell .markdown}
Способ инициализации точечного источника c энергией En и распр. ВФ на
расстоянии z от него
:::

::: {.cell .code execution_count="2"}
``` python
p = od.Point_source(z=100, En=10) 
E_arr = p.E()
```
:::

::: {.cell .markdown}
Пример инициализации пластиковых (C21SO8SiH36) линз (идеальная и
неидеальная)
:::

::: {.cell .code execution_count="3"}
``` python
# N1_global = 10
# Copy_flag = False
# arr_len = 2 if Copy_flag else 2*N1_global

# phases = 2*np.pi*np.random.rand(arr_len)
# w_s = (np.random.rand(arr_len)-0.5)*np.pi/180*1
# s_s  = (np.random.rand(arr_len)-0.5)*2e-6*1

# crl = od.CRL(lam=p.lam, arr_start=E_arr,\
#                 R=5e-6, A=24e-6, d=5e-6, N1=N1_global, z=0,\
#                     molecula="C21SO8SiH36", density=1.12, Flen=0, gap=0)

# crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
#                     R=5e-6, A=24e-6, d=5e-6, N1=N1_global, z=0,\
#                         molecula="C21SO8SiH36", density=1.12, Flen=0, gap=0,\
#                             b=1e-6, m=3e6, copy=Copy_flag, arr_phase=phases, arr_s=s_s, arr_w=w_s)
                                
```
:::

::: {.cell .markdown}
Пример инициализации кремниевых (Si) линз (идеальная и неидеальная)
:::

::: {.cell .code execution_count="21"}
``` python
N1_global = 100
Copy_flag = True
arr_len = 2 if Copy_flag else 2*N1_global

phases = 2*np.pi*np.random.rand(arr_len)*0
w_s = (np.random.rand(arr_len)-0.5)*np.pi/180*0
s_s  = (np.random.rand(arr_len)-0.5)*2e-6*0

crl = od.CRL(lam=p.lam, arr_start=E_arr,\
                R=6.25e-6, A=50e-6, d=2e-6, N1=N1_global, z=0,\
                    molecula="Si", density=2.33, Flen=0, gap=0)

crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
                    R=6.25e-6, A=50e-6, d=2e-6, N1=N1_global, z=0,\
                        molecula="Si", density=2.33, Flen=0, gap=0,\
                            b=2e-7, m=3e8, copy=Copy_flag, arr_phase=phases, arr_s=s_s, arr_w=w_s)
```
:::

::: {.cell .markdown}
!!!Важно!!! Подсчёт и установка фокусного расстояния для линз
:::

::: {.cell .code execution_count="22"}
``` python
focus = crl.focus()
crl.set_z(z=focus)
crlm.set_z(z=focus)
```
:::

::: {.cell .markdown}
Распределение интенсивности излучения в фокусе
:::

::: {.cell .code execution_count="23"}
``` python
plt.plot(p.x, crl.I(), label="numerical CRL")
plt.plot(p.x, crlm.I(), label="numerical CRLm")
plt.legend()
plt.xlim(-5e-7, 5e-7)
plt.grid()
plt.show()
```

::: {.output .stream .stderr}
    100%|██████████| 99/99 [00:00<00:00, 607.29it/s]
    100%|██████████| 99/99 [00:00<00:00, 775.54it/s]
:::

::: {.output .display_data}
![](vertopal_49f43c458a9e444da55782d0fb585588/99226dcfadbb3ee16ada1870408ca2179b6f38d6.png)
:::
:::
