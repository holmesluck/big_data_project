import tensorflow as tf
import pandas as pd

read_data_pd_training = pd.read_csv('./letter_recognition_training_data_set.csv')
read_data_pd_testing = pd.read_csv('./letter_recognition_testing_data_set.csv')
ohv_data = pd.get_dummies(read_data_pd_training,sparse=True)

#training data set
train_data = ohv_data.iloc[:16000,:16]
train_label = ohv_data.iloc[:16000,-26:]
#training length
train_length = len(train_data)

#simulate testing data set
test_data = ohv_data.iloc[-4000:,:16]
test_label = ohv_data.iloc[-4000:,-26:]



# Parameters
learning_rate = 0.008
training_epochs = 15
batch_size = 1000
display_step = 100

# Network Parameters
n_hidden_0 = 256 # 0 layer
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 16 # data input
n_classes = 26 #total classes (A-Z letters)




# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    #Hidden layer with Sigmod avtivation
    layer_0 = tf.add (tf.matmul (x, weights['h0']), biases['b0'])
    layer_0 = tf.nn.sigmoid(layer_0)
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(layer_0, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h0': tf.Variable(tf.random_normal([n_input, n_hidden_0])),
    'h1': tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b0': tf.Variable(tf.random_normal([n_hidden_0])),
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model

pred = multilayer_perceptron(x, weights, biases)


# Define loss and optimizer
test_prediction = tf.nn.softmax(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_length/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: train_data.iloc[i*batch_size:(i+1)*batch_size-1],
                                                          y: train_label.iloc[i*batch_size:(i+1)*batch_size-1]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    print(pred)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #for each capital letters accuracy value
    print ("Accuracy A:",accuracy.eval({x: test_data, y: test_label.iloc[-4000:,:1]}))
    print ("Accuracy B:",accuracy.eval ({x: test_data, y: test_label.iloc[-4000:,1:2]}))

    print("Accuracy:", accuracy.eval({x: test_data, y: test_label}))

