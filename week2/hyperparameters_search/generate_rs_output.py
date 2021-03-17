
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy
from numpy.random import randn
from scipy import array, newaxis
from tqdm import tqdm

    
import imageio
import cv2

df = pd.read_csv("random_search_finer.csv")
df["alpha"] = df["alpha"].apply(lambda p: p.replace("[", "").replace("]", "")).astype(float)
df["p"] = df["p"].astype(float)
df = df.sort_values(["alpha", "p"], ascending=True)


alphas = list(df["alpha"])
ps = list(df["p"])
maps = list(df["mAP"])

Xs = np.array(alphas)#np.arange(1, 25, 1)
Ys = np.array(ps) #np.arange(0.001, 0.2, 0.003)
Zs = np.array(maps)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(Xs, Ys, Zs, cmap="Reds", linewidth=0)
ax.set_xlabel("alpha")
ax.set_ylabel("p")

fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

STEP = 1
# rotate the axes and update
for angle in tqdm(range(0, 360, STEP)):
    ax.view_init(30, angle)
    #plt.pause(.001)
    plt.savefig(f"results/{angle}.png", dpi=60)
    
    
print("GENERATED! Building gif...")

# rotate the axes and update
imgs = []
for angle in tqdm(range(0, 360, STEP)):
    img = cv2.cvtColor(cv2.imread(f"results/{angle}.png"), cv2.COLOR_BGR2RGB)
    imgs.append(img)
    
imageio.mimsave(f'results.gif', imgs)