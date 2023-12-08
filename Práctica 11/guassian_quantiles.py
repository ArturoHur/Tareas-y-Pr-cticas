from sklearn.datasets import make_gaussian_quantiles
import numpy as np

N = 1000 # muestras
gaussian_quantiles = make_gaussian_quantiles(mean=None,
                        cov=0.1,
                        n_samples=N,
                        n_features=2,
                        n_classes=2,
                        shuffle=True,
                        random_state=None)

X, Y = gaussian_quantiles
Y = Y[:,np.newaxis]