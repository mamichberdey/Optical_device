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
Тестовая реализация кода с помощью opticaldevicelib
:::

::: {.cell .code execution_count="2"}
``` python
import time
import opticaldevicelib as od
od.Optical_device.set_value(new_dx=5e-8, new_N=2**15)
```
:::
