import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



X = np.array([[-5, 10,2], [-2, -3,3], [-4, -9,1], [7, 11,-3], [12, 6,-1], [13, 4,5]])

pca = PCA(n_components=1)
pca_transf = pca.fit_transform(X)

print (pca_transf)