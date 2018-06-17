import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import warnings

network_input = np.load('./NumpyDataset/Input-Tensor.npy')
network_output = np.load('./NumpyDataset/Output-Tensor.npy')
print(network_input.shape, network_output.shape)

time_steps=100

print('Creating graph')

x = tf.placeholder(dtype=tf.float32, 
                       shape=(None, 100, 1))
    
y_true = tf.placeholder(dtype=tf.float32, 
                       shape=(None, 358))
    
y_true_cls = tf.argmax(y_true, axis=1)
    
inpt=tf.unstack(x ,time_steps,1)
num_units = [256, 512, 256]
rnn_layers = []
    
for size in num_units:
        
    rnn_layers.append(tf.nn.rnn_cell.LSTMCell(size, 
                                              forget_bias=1)) 
        
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    
outputs, states = tf.nn.static_rnn(cell=multi_rnn_cell, 
                                   inputs=inpt,
                                   dtype=tf.float32)
    
output_dense1 = tf.layers.dense(outputs[-1], 
                                    256, 
                                    activation=None)
    
logits = tf.layers.dense(output_dense1, 
                         358, 
                         activation=None)
    
y_pred = tf.nn.softmax(logits=logits)
    
y_pred_cls = tf.argmax(y_pred, 
                       axis=1)
    
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, 
                                                           logits=logits)
loss = tf.reduce_mean(cross_entropy)
    
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss)
    
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Built graph sucessfully')

num_iterations = 1783
batch_size = 32
epochs = 10
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
def optimize():
    
    for i in range(num_iterations):
            
        x_true_batch = network_input[i*batch_size:(i*batch_size)+batch_size]
        y_true_batch = network_output[i*batch_size:(i*batch_size)+batch_size]

        feed_dict = {x: x_true_batch,
                     y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 20 == 0:
	        acc = sess.run(accuracy,feed_dict=feed_dict)
	        print('Accuracy after %d iterations : %.7f' % (i+1, acc))
	        saver.save(sess, './model-checkpoints/saved_model', global_step=20)
	        print('Model saved')
        
print('Starting Training')
for j in range(epochs):
    optimize()
    print('%d Epochs completed' % (j+1))