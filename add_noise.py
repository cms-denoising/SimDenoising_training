import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_bin_weights(file, n):
    tree = file["g4SimHits/tree"]
    branch = tree.array("bin_weights")
    data = np.zeros((100,100))
    count = 0
    for y in range(100):
        for x in range(100):
            data[99-x][y]=branch[n][count]
            count+=1
    return data;
    

def add_noise(data, sigma):
    noisy = data + np.random.normal(loc=0.0,scale=sigma, size=[100,100])
    noisy = np.clip(noisy, a_min=0, a_max=None)
    return noisy;

"""
#example
file = uproot.rootio.open("test.root")
data = get_bin_weights(file, 0)
noisy = add_noise(data,0.5)
"""


