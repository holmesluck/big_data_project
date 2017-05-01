#test the one-hot-vector
import numpy as np
import pandas as pd
import theano
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

read_data_pd_training = pd.read_csv('./letter_recognition_training_data_set.csv')
#use the one hot vector to do the data preprocess
# enc = OneHotEncoder()
# enc.fit(read_data_pd_training)
# data = enc.transform(read_data_pd_training).toarray()
# print (data)
# # data = read_data_pd_training.T.to_dict().values()
# # vectorizer = DV( sparse = False)
# x = theano.tensor.vector()
# theano.tensor.extra_ops.to_one_hot(y, nb_class, dtype=None)
# read_data_pd_training.label = read_data_pd_training.label.astype(float)
# read_data_pd_training = read_data_pd_training.select(col('label'),read_data_pd_training.label.cast(''))
label_data = pd.get_dummies(read_data_pd_training,sparse=True)

# for i in range(len(label_data)):
print (type(label_data))
# print (label_data.icol(i))