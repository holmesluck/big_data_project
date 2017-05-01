import pandas as pd


read_data_pd_training = pd.read_csv('./letter_recognition_training_data_set.csv')
label_data = pd.get_dummies(read_data_pd_training,sparse=True)
# drop_features = ['','']
# label_data.drop(drop_features, axis=1)
# print (label_data)
train_length = len(read_data_pd_training)

train_data = label_data.iloc[:16000,:16]
train_label = label_data.iloc[:16000,-26:]

test_data = label_data.iloc[-4000:,:16]
test_label = label_data.iloc[-4000:,-26:]



print('test_label')
print(test_label.iloc[-4000:,:1])