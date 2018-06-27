import tensorflow as tf
import numpy as np
import warnings
import random
warnings.filterwarnings("ignore")

tf.reset_default_graph()

network_input = np.load('./NumpyDataset/Input-Tensor.npy')
network_output = np.load('./NumpyDataset/Output-Tensor.npy')
time_steps=100
n_classes = 358
epochs = 100
num_units = 256

print('Creating graph')

x = tf.placeholder(dtype=tf.float32, 
                       shape=(None, 100, 1))
    
y_true = tf.placeholder(dtype=tf.float32, 
                       shape=(None, n_classes))
    
y_true_cls = tf.argmax(y_true, axis=1)
    
inpt=tf.unstack(x ,time_steps,1)

lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1)
    
outputs, states = tf.nn.static_rnn(cell=lstm_layer, 
                                   inputs=inpt,
                                   dtype=tf.float32)

output_dense1 = tf.layers.dense(outputs[-1], 
                                    512, 
                                    activation=tf.nn.relu)
dropout1 = tf.layers.dropout(output_dense1, 0.7)
output_dense2 = tf.layers.dense(dropout1,
								256,
								activation=tf.nn.relu)
logits = tf.layers.dense(output_dense2,
						 n_classes,
						 activation=None)
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
batch_size = 64
num_iterations = 891
epochs = 10
save_every = 20

def optimize():
    
    for i in range(num_iterations):
            
        x_true_batch = network_input[i*batch_size:(i*batch_size)+batch_size]
        y_true_batch = network_output[i*batch_size:(i*batch_size)+batch_size]

        feed_dict = {x: x_true_batch,
                     y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict)
        if i % save_every == 0 and i!=0:
        	j = random.randint(0, 1783)
        	feed_dict_acc = {x: network_input[j*batch_size:(j*batch_size)+batch_size],
    					 	 y_true: network_output[j*batch_size:(j*batch_size)+batch_size]}
	        acc, y_p, y_t = sess.run([accuracy, y_pred_cls, y_true_cls],feed_dict=feed_dict_acc)
	        print('Accuracy after %d iterations : %.7f' % (i+1, acc))
	        print(y_p)
	        print(y_t)

	        # saver.save(sess, './model-checkpoints/saved_model')
	        # print('Model saved')

        
print('Starting Training')
for j in range(epochs):
    optimize()
    print('%d Epochs completed' % (j+1))

