import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

def getxywh():

    coord = []
    n = torch.randint(10, 100, [1])
    for i in range(n):
        (x, y) = torch.randint(50, 100, [2])
        coord.append((x, y))
        plt.scatter(x.numpy(), y.numpy())

    print(len(coord))
    plt.show()

if __name__ == '__main__':
    getxywh()