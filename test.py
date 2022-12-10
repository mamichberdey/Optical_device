
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


x = np.array([1,3,6,8,12,25])[::-1]
# mask = abs(x)<=20 

# np.ma.fix_invalid(x[, mask, copy=False, fill_value=0])
# print(np.ma.flatnotmasked_edges(mask))
# x_mask = x if (mask==True).any() else 0
print(-(np.diff(x)))
# print(x_mask)