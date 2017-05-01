#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 07:34:01 2017

### Lab 2 ###

@author: lizhuo

"""
### import libraries ###
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xlsxwriter



###############################################################################

### functions ###
def feature_scaling(input_pd, scaling_meathod):
    print ('*** feature scaling ***')
    # scaled_pd = pd.DataFrame()
    if scaling_meathod == 'z-score':
        scaled_pd = (input_pd - input_pd.mean()) / input_pd.std()
    elif scaling_meathod == 'min-max':
        scaled_pd = (input_pd - input_pd.min()) / (input_pd.max() - input_pd.min())
    return scaled_pd


def input_reshape(input_pd, start, end, batch_size, batch_shift, n_features):
    print ('*** input reshape ***')
    temp_pd = input_pd[start - 1: end + batch_size - 1]
    output_pd = map(lambda y: temp_pd[y:y + batch_size], range(0, end - start + 1, batch_shift))
    output_temp = map(lambda x: np.array(output_pd[x]).reshape([-1]), range(len(output_pd)))
    output = np.reshape(output_temp, [-1, batch_size, n_features])
    return output


def target_reshape(input_pd, start, end, batch_size, batch_shift, n_step_ahead, m_steps_pred):
    print ('*** target reshape ***')
    temp_pd = input_pd[start + batch_size + n_step_ahead - 2: end + batch_size + n_step_ahead + m_steps_pred - 2]
    output_pd = map(lambda y: temp_pd[y:y + m_steps_pred], range(0, end - start + 1, batch_shift))
    output_temp = map(lambda x: np.array(output_pd[x]).reshape([-1]), range(len(output_pd)))
    output = np.reshape(output_temp, [-1, 1])
    return output


def lstm(input, n_inputs, n_steps, n_of_layers, scope_name):
    num_layers = n_of_layers
    input = tf.transpose(input, [1, 0, 2])
    input = tf.reshape(input, [-1, n_inputs])
    input = tf.split(input, n_steps, 0)
    with tf.variable_scope(scope_name):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_inputs)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        output, state = tf.contrib.rnn.static_rnn(cell, input, dtype=tf.float32)
    output = output[-1]
    return output

#array for different variable
iterations_t = []
validation_mse_t = []
validation_rmse_t = []
validation_mae_t = []
validation_mape_t = []
prediction_mse_t = []
prediction_rmse_t = []
prediction_mae_t = []
prediction_mape_t = []

# get the value from the iterations module
def get_i_and_8_value(ii,validation_mse,validation_rmse,validation_mae,validation_mape
                    ,prediction_mse,prediction_rmse,prediction_mae,prediction_mape):

    iterations_t.append(ii)
    validation_mse_t.append(validation_mse)
    validation_rmse_t.append(validation_rmse)
    validation_mae_t.append(validation_mae)
    validation_mape_t.append(validation_mape)
    prediction_mse_t.append(prediction_mse)
    prediction_rmse_t.append(prediction_rmse)
    prediction_mae_t.append(prediction_mae)
    prediction_mape_t.append(prediction_mape)

# change the batch size
def switch_bs(a):
    if a==1:
     batch_size = 100
     return batch_size
    else:
     batch_size = 50
     return batch_size
# change the learing rate
def switch_lr(b):
    if b == 0:
     learning_rate = 0.1
     return learning_rate
    if b == 1:
     learning_rate = 0.01
     return learning_rate
    else:
     learning_rate = 0.001
     return learning_rate
### parameter setting ###
# the selected input features
feature_to_input = ['open price', 'highest price', 'lowest price', 'close price', 'volumn']
# the item we want to predict
feature_to_predict = ['close price']
# the feature needed to do feature scaling
feature_to_scale = ['volumn']
# feature scaling method
sacling_meathod = 'min-max'

# specifify the start and end position of training and testing
train_start = 1
train_end = 1200
test_start = 1201
test_end = 1400

# the length of a batch
batch_size = 50
# the interval between two batches
batch_shift = 1
# the intervals between the training end and the forecast day
n_step_ahead = 1
m_steps_pred = 1
# how many features that will be input
n_features = len(feature_to_input)

# lstm parameters
# name of the lstm function
lstm_scope_name = 'lstm_prediction'
# number of lstm layers
n_lstm_layers = 1

### regression parameter
n_pred_class = 1

### learning parameter
learning_rate = 0.01

# iterative parameters
# number of iterations
EPOCHS = 5000
# the interval of printing validation result
PRINT_STEP = np.array([1,5,10,20,50,100,200,500,1000,2000,5000])

### read data and data preprocessings ###
# read .csv file by pd.read_csv command, and the results will be stored as pandas DataFrame
read_data_pd = pd.read_csv('./stock_price.csv')
# drop the column 'year', 'month', 'day', in other words, keep the column 'open prince', 'highest price', 'lowest price' and 'close price'
input_pd = read_data_pd.drop(['year', 'month', 'day'], axis=1)
# call the function feature scaling
temp_pd_1 = input_pd[feature_to_input]
temp_pd_2 = input_pd[feature_to_predict]
temp_pd = feature_scaling(input_pd[feature_to_scale], sacling_meathod)
input_pd[feature_to_scale] = temp_pd

### construct the training data ###
train_input_temp_pd = input_pd[feature_to_input]
# call the 'input_reshape' function to construct the training input data (a matrix)
train_input_nparr = input_reshape(train_input_temp_pd, train_start, train_end, batch_size, batch_shift, n_features)

train_target_temp_pd = input_pd[feature_to_predict]
# call the 'target_reshape' function to construct the training target data (label) (a column vector)
train_target_nparr = target_reshape(train_target_temp_pd, train_start, train_end, batch_size, batch_shift, n_step_ahead,
                                    m_steps_pred)

### construct the testing data ###
test_input_temp_pd = input_pd[feature_to_input]
# call the 'input_reshape' function to construct the training input data (a matrix)
test_input_nparr = input_reshape(test_input_temp_pd, test_start, test_end, batch_size, batch_shift, n_features)

test_target_temp_pd = input_pd[feature_to_predict]
# call the 'target_reshape' function to construct the training target data (label) (a column vector)
test_target_nparr = target_reshape(test_target_temp_pd, test_start, test_end, batch_size, batch_shift, n_step_ahead,
                                   m_steps_pred)



### construct the training data without feature scaling  ###
train_input_temp_pd_1 = temp_pd_1
# call the 'input_reshape' function to construct the training input data (a matrix)
train_input_nparr_1 = input_reshape(train_input_temp_pd_1, train_start, train_end, batch_size, batch_shift, n_features)

train_target_temp_pd_1 = temp_pd_2
# call the 'target_reshape' function to construct the training target data (label) (a column vector)
train_target_nparr_1 = target_reshape(train_target_temp_pd_1, train_start, train_end, batch_size, batch_shift, n_step_ahead,
                                    m_steps_pred)

### construct the testing data ###
test_input_temp_pd_1 = temp_pd_1
# call the 'input_reshape' function to construct the training input data (a matrix)
test_input_nparr_1 = input_reshape(test_input_temp_pd_1, test_start, test_end, batch_size, batch_shift, n_features)

test_target_temp_pd_1 = temp_pd_2
# call the 'target_reshape' function to construct the training target data (label) (a column vector)
test_target_nparr_1 = target_reshape(test_target_temp_pd_1, test_start, test_end, batch_size, batch_shift, n_step_ahead,
                                   m_steps_pred)





######### main body #########
# reset the graph
tf.reset_default_graph()

# difine two place holders, which can be reagarded as handles pointing to a memory space,
# but actually the memory has not been allocated to it before start the session.

# x_ has the size of 3-dimension, None means the 1st dimension can be of any length,
# the 2nd and 3rd dimension are of size batch_size and n_features
x_ = tf.placeholder(tf.float32, [None, batch_size, n_features])
# the same as x_ but it is a placeholder of 2-dimension
y_ = tf.placeholder(tf.float32, [None, 1])

### call the lstm-rnn function ###
lstm_output = lstm(x_, n_features, batch_size, n_lstm_layers, lstm_scope_name)

### linear regressor ###
# w is the weight vector
W = tf.Variable(tf.random_normal([n_features, n_pred_class]))
# b is the bias
b = tf.Variable(tf.random_normal([n_pred_class]))
# Y = WX + b
y = tf.matmul(lstm_output, W) + b

### define the cost function (loss function)
cost_func = tf.reduce_mean(tf.square(y - y_))

### define the learning algorithm (optimizer), such as gradient descent,
# stochastic gradient, RMSP, Adam.

# train_op = tf.train.RMSPropOptimizer(0.001, 0.2).minimize(cost)
# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_func)

### start the session() ###

# Calculate MSE, RMSE, MAE and MAPE, and present them in a table
# creat a document called result/xlsx to store the data in a table
workbook  = xlsxwriter.Workbook('Result.xlsx')
# create a worksheet
worksheet = workbook.add_worksheet('All_data_result')
# write the title at the A1 cell.
worksheet.write('A1','The result of the lab 2')
# initial the row and colum
row = 0
col = 0

# initialize all variables
# for a in range(2):
#  switch_bs(a)
#  print('*** this is the batch size equal to'+' '+str(switch_bs(a))+'***')
# for b in range(3):
#  switch_lr(b)
#  print('*** this is the learning rate equal to'+' '+ str (switch_lr(b)) + '***')
init = tf.global_variables_initializer()
with tf.Session() as sess:
   sess.run(init)
   for c in range(2):
    for ii in range(EPOCHS+1):
        for i in range(len(PRINT_STEP)):
         if c==0:
          sess.run(train_op, feed_dict={x_: train_input_nparr, y_: train_target_nparr})

          if ii == PRINT_STEP[i]:

            cost = sess.run(cost_func, feed_dict={x_: train_input_nparr, y_: train_target_nparr})
            print ('iteration ='+str(ii)+','+'training cost:'+str(cost))

            print ('-------------- validation ----------------')
            validation = sess.run(y, feed_dict={x_: train_input_nparr})
            validation_mse = np.mean(np.square(validation - train_target_nparr))
            validation_rmse = np.sqrt(np.mean(np.square(validation - train_target_nparr)))
            validation_mae = np.mean(np.abs(validation - train_target_nparr))
            validation_mape = np.mean(np.abs((validation - train_target_nparr) / train_target_nparr))



            print ('iterations'+str(ii))
            print ('validation_mse ='+str(validation_mse))
            print ('validation_rmse ='+str(validation_rmse))
            print ('validation_mae ='+str(validation_mae))
            print ('validation_mape ='+ str(validation_mape))

            print ('-------------- prediction ----------------')
            prediction = sess.run(y, feed_dict={x_: test_input_nparr})
            prediction_mse = np.mean(np.square(prediction - test_target_nparr))
            prediction_rmse = np.sqrt(np.mean(np.square(prediction - test_target_nparr)))
            prediction_mae = np.mean(np.abs(prediction - test_target_nparr))
            prediction_mape = np.mean(np.abs((prediction - test_target_nparr) / test_target_nparr))

            print ('prediction_mse ='+str(prediction_mse))
            print ('prediction_rmse ='+ str(prediction_rmse))
            print ('prediction_mae ='+str(prediction_mae))
            print ('prediction_mape ='+str(prediction_mape))
            print ('')
            print ('')

            get_i_and_8_value(ii,validation_mse,validation_rmse,validation_mae,validation_mape
                              ,prediction_mse,prediction_rmse,prediction_mae,prediction_mape)
        # print the fianl result in interations equal to 5000
          if ii == 5000:
            print ('-------------- validation ----------------')
            validation = sess.run(y, feed_dict={x_: train_input_nparr})
            validation_mse = np.mean(np.square(validation - train_target_nparr))
            validation_rmse = np.sqrt(np.mean(np.square(validation - train_target_nparr)))
            validation_mae = np.mean(np.abs(validation - train_target_nparr))
            validation_mape = np.mean(np.abs((validation - train_target_nparr) / train_target_nparr))

            worksheet.write(row, col, 'validation_mse')
            worksheet.write(row, col + 1, 'validation_rmse')
            worksheet.write(row, col + 2, 'validation_mae')
            worksheet.write(row, col + 3, 'validation_mape')
            worksheet.write(row, col + 4, 'prediction_mse')
            worksheet.write(row, col + 5, 'prediction_rmse')
            worksheet.write(row, col + 6, 'prediction_mae')
            worksheet.write(row, col + 7, 'prediction_mape')

            worksheet.write(row + 1, col, str(validation_mse))
            worksheet.write(row + 1, col + 1, str(validation_rmse))
            worksheet.write(row + 1, col + 2, str(validation_mae))
            worksheet.write(row + 1, col + 3, str(validation_mape))

            print ('validation_mse ='+str(validation_mse))
            print ('validation_rmse ='+str(validation_rmse))
            print ('validation_mae ='+str(validation_mae))
            print ('validation_mape ='+str(validation_mape))

            print ('-------------- prediction ----------------')
            prediction = sess.run(y, feed_dict={x_: test_input_nparr})
            prediction_mse = np.mean(np.square(prediction - test_target_nparr))
            prediction_rmse = np.sqrt(np.mean(np.square(prediction - test_target_nparr)))
            prediction_mae = np.mean(np.abs(prediction - test_target_nparr))
            prediction_mape = np.mean(np.abs((prediction - test_target_nparr) / test_target_nparr))

            worksheet.write(1, col + 4, str(prediction_mse))
            worksheet.write(1, col + 5, str(prediction_rmse))
            worksheet.write(1, col + 6, str(prediction_mae))
            worksheet.write(1, col + 7, str(prediction_mape))

            print ('prediction_mse ='+str(prediction_mse))
            print ('prediction_rmse ='+str(prediction_rmse))
            print ('prediction_mae ='+str(prediction_mae))
            print ('prediction_mape ='+str(prediction_mape))
            # draw the picture
            plt.figure (1)
            plt.plot (train_target_nparr, color='blue', linewidth=1.5)
            plt.plot (validation, color='green', linewidth=1.5)
            plt.title ('Compare the Real Stock Price and the Validation Results', fontsize=18)
            plt.xlabel ('Day Index', fontsize=17)
            plt.ylabel ('Price', fontsize=17)
            plt.legend (['real price', 'validation'], fontsize=17)
            plt.savefig ('/Users/zhangyangzuo/Downloads/capture/validation_value.png')
            plt.show ()

            plt.figure (2)
            plt.plot (test_target_nparr, color='blue', linewidth=1.5)
            plt.plot (prediction, color='red', linewidth=1.5)
            plt.title ('Compare the Real Stock Price and the Prediction Results', fontsize=18)
            plt.xlabel ('Day Index', fontsize=17)
            plt.ylabel ('Price', fontsize=17)
            plt.legend (['real price', 'prediction'], fontsize=17)
            plt.savefig ('/Users/zhangyangzuo/Downloads/capture/prediction_value.png')
            plt.show ()


# close the work sheet
workbook.close()



# plot the figure iterations with MSE,RMSE,MAE,MAPE
plt.figure(3)

plt.plot(iterations_t,validation_mse_t,color='green',linewidth=1.5)

plt.plot(iterations_t,validation_rmse_t ,color='red',linewidth=1.5)

plt.plot(iterations_t,validation_mae_t ,color='peru',linewidth=1.5)

plt.plot(iterations_t,validation_mape_t ,color='mistyrose',linewidth=1.5)

plt.plot(iterations_t,prediction_mse_t,color='sienna',linewidth=1.5)

plt.plot(iterations_t,prediction_rmse_t ,color='khaki',linewidth=1.5)

plt.plot(iterations_t,prediction_mae_t,color='lime',linewidth=1.5)

plt.plot(iterations_t,prediction_mape_t,color='mediumpurple',linewidth=1.5)

plt.title('Relation between the Iterations and MSE, RMSE, MAE, MAPE', fontsize=18)
plt.xlabel('Iterations',fontsize=17)
plt.ylabel('Different Varibles',fontsize=17)
plt.legend(['validation_mse','validation_rmse','validation_mae','validation_mape',
            'prediction_mse','prediction_rmse','prediction_mae','prediction_mape'],fontsize=10)
plt.savefig('/Users/zhangyangzuo/Downloads/capture/multipleline_value.png')
plt.show()




