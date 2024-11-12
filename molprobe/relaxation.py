import numpy as np

def find_relax_time(t_array, autocov, crit=np.exp(-1), estimate=False):
    index = 0
    for j, elt in enumerate(autocov):
        if elt < crit:
            index = j
            break
    
    if index != 0:
        x1, x2 = t_array[index], t_array[index-1]
        y1, y2 = autocov[index], autocov[index-1]
        a = (y2 - y1) / np.log10(x2/x1)
        b = y2 - a * np.log10(x2)
        t_relax = 10**((crit - b) / a)
    
    elif estimate:
        t_relax = 10**9
    else:
        t_relax = np.nan
    return t_relax