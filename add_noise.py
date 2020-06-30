import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import random 

def get_bin_weights(file, n):
    tree = file["g4SimHits/tree"]
    branch = tree.array("bin_weights")
    data = np.zeros((100,100))
    count = 0
    for y in range(100):
        for x in range(100):
            data[99-x][y]=branch[n][count]
            count+=1
    # do random rotation/flips
    flipx = random.randint(0, 1)
    flipy = random.randint(0, 1)
    rot = random.randint(0, 3)
    if (flipx):
        data = np.fliplr(data)
    if flipy:
        data = np.flipud(data)
    for i in range(rot):
        data = np.rot90(data)
    return data;
    

def add_noise(data, sigma):
    return np.clip(data + np.random.normal(loc=0.0,scale=sigma, size=[100,100]), a_min=0, a_max=None);


#example
if __name__=="__main__":
    file = uproot.rootio.open("test.root")
    truth = get_bin_weights(file, 1)
    noisy = add_noise(truth,0.5)
    plt.imshow(truth)
    plt.savefig("truth.png")
    plt.imshow(noisy)
    plt.savefig("noisy.png")

