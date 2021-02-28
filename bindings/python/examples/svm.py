import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from ado import svm
import random

random.seed(16)
np.random.seed(16)


n_samples = 40
X, y = make_blobs(n_samples=n_samples, centers=2,
                  random_state=16, cluster_std=2.0)
y[y == 0] = -1

# Define the optimization problem.
kernel_list = ["linear", "rbf"]

plt.figure()

for idx, k_type in enumerate(kernel_list):
    plt.subplot(1, 2, idx + 1)
    plt.title(str(k_type))

    opt = svm(1.0, 1e-4, k_type, 1000, 16, 5)
    opt.fit(X, y)

    margin = 2
    k = np.linspace(np.min(X) - margin, np.max(X) + margin, num=n_samples)
    xx, yy = np.meshgrid(k, k)
    K = np.stack([xx.flatten(), yy.flatten()], axis=1)
    M = opt.predict(K).reshape((len(k), len(k)))

    plt.imshow(M, extent=[np.min(X) - margin, np.max(X) + margin, np.min(X) - margin, np.max(X) + margin],
               origin='lower',
               cmap='inferno')
    plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap='winter')


plt.show()
