import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


X = np.array([[-5, 10,2], [-2, -3,3], [-4, -9,1], [7, 11,-3], [12, 6,-1], [13, 4,5]])


scaled_x=np.mean(X[:,0])-X[:,0]
scaled_y=X[:,1]-np.mean(X[:,1])
scaled_z=X[:,2]-np.mean(X[:,2])

# #combine two 1-d arrays into one 2-d array
data=np.vstack((scaled_x,scaled_y,scaled_z)).T
scatter_matrix=np.dot(np.transpose(data),data)

# #Calculate the eigenvectors and eigenvalues for scatter matrix.
eig_val, eig_vec = np.linalg.eig(scatter_matrix)
# #On top of the data we plot both the eigenvectors
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],'bo-',label='x',color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],'ys-',label ='y',color='blue')
plt.plot([eig_vec[:,2][0],0],[eig_vec[:,2][1],0],'mD-',label='z',color='green')
# #Choose components and form a feature vector
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature=eig_pairs[0][1]
# # Deriving the new data set
new_data_reduced=np.dot(data,np.transpose(feature))
print("data from the PCA by myself")
print (new_data_reduced)
#draw the new data set
plt.plot(scaled_x,[0.5]*6,'o',label ='x',color='red')
plt.plot(scaled_y,[0.5]*6,'o',label ='y',color='blue')
plt.plot(scaled_z,[0.5]*6,'o',label ='z',color='green')
plt.plot(new_data_reduced,[0.5]*6,'*',label ='PCA data set',color='purple')

# use PCA from sklearn to verify the results
pca = PCA(n_components=1)
pca_transf = pca.fit_transform(X)

print ("data from call the method PCA from sklearn")
print (pca_transf)





plt.legend(loc="lower right")
plt.show()