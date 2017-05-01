import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9])

scaled_x=x-x.mean()
scaled_y=y-y.mean()

#combine two 1-d arrays into one 2-d array
data=np.vstack((scaled_x,scaled_y)).T
scatter_matrix=np.dot(np.transpose(data),data)

#Calculate the eigenvectors and eigenvalues for scatter matrix.
eig_val, eig_vec = np.linalg.eig(scatter_matrix)
#On top of the data we plot both the eigenvectors
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='blue')
#Choose components and form a feature vector
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature=eig_pairs[0][1]
# Deriving the new data set
new_data_reduced=np.dot(data,np.transpose(feature))

#draw the new data set
pca = PCA(n_components=1)
pca_transf = pca.fit_transform(data)
print ("pca")
print (pca_transf)
# print("data")
# print (data)
# print("scatter")
# print (scatter_matrix)
print("new data")
print (new_data_reduced)
plt.plot(new_data_reduced,[1.2]*10,'*',color='green')
plt.scatter(scaled_x, scaled_y,color='red')
plt.show()