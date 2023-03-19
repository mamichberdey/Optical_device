import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(os.path.abspath("output/3d_peak_output.csv"), sep=";")
j_arr = df["j"].to_numpy(dtype=float)
df.drop(df.columns[[0]], axis=1, inplace=True)
i_arr = df.columns.to_numpy(dtype=float)


# print(j_arr*1e6)

X, Y = np.meshgrid(i_arr*1e-6, j_arr*1e6)
# X = np.log10(X)
# Y = np.log10(Y)

Z = df.to_numpy()

ax = plt.axes(projection='3d')
ax.set_xlabel(r'freq, µ$m^{-1}$', labelpad=20)
ax.set_ylabel(r'amplitude, µm', labelpad=20)
ax.set_zlabel('Max Intensity', labelpad=20)
ax.plot_surface(X,Y,Z,  cmap ='inferno')
ax.set_title(r'Зависимость максимальной интенсивности от b, m')
plt.show()


# fig,ax=plt.subplots(1,1)
# cp = ax.pcolor(X, Y, Z, cmap="inferno")
# # plt.yscale('log')
# # plt.xscale('log')
# fig.colorbar(cp) # Add a colorbar to a plot
# # ax.set_title(r'Зависимость максимальной интенсивности от b, m')
# ax.set_xlabel(r'частота, мк$м^{-1}$')
# ax.set_ylabel(r'амплитуда, мкм')
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# # ax.set_xlim(i_arr[0]*1e-6, i_arr[-1]*1e-6)
# # ax.set_ylim(j_arr[0]*1e6, j_arr[-1]*1e6)
# # ax.set_zlabel('Max Intensity', labelpad=20)
# # plt.savefig("fig1", dpi=800)
# plt.show()