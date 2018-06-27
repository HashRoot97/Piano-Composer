import tensorflow as tf
import numpy as np

def load_batch(iter):

  b_x_ = network_input[iter*batch_size:iter*batch_size + batch_size]
  b_y_true = network_output[iter*batch_size:iter*batch_size + batch_size]

  b_x = []
  for i in range(b_x_.shape[0]):
    tmp = []
    for j in range(b_x_.shape[1]):
      zer = np.zeros((n_classes,))
      zer[b_x_[i][i]] = 1
      tmp.append(zer)
    b_x.append(tmp)
  b_x = np.asarray(b_x, dtype=np.float32)
  return b_x, b_y_true

network_input = np.load('./NumpyDataset/Input-Tensor.npy')
network_output = np.load('./NumpyDataset/Output-Tensor.npy')
time_steps=100
n_classes = 358
epochs = 10
num_units = 256
batch_size = 16
num_iterations = 3567
hidden_size = 358
num_layers = 2
acc_every = 10

net_x, net_y_true = load_batch(1)

print('Defining Model')

x = tf.placeholder(tf.float32, [None, time_steps, n_classes])
y_true = tf.placeholder(tf.float32, [None, n_classes])
y_true_cls = tf.argmax(y_true, axis=1)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

initial_state = stacked_lstm.zero_state(batch_size,  tf.float32)

outp, next_state = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=initial_state)

outputs = tf.reshape(outp[:, -1:, :], [batch_size, 358])

print('After Stacked LSTM : ', outputs)

output_dense1 = tf.layers.dense(outputs, 
                                    512, 
                                    activation=tf.nn.relu)
print('After Dense 1 : ', output_dense1)

dropout1 = tf.layers.dropout(output_dense1, 0.7)
output_dense2 = tf.layers.dense(dropout1,
                256,
                activation=tf.nn.relu)

print('After Dense2 : ', output_dense2)

logits = tf.layers.dense(output_dense2,
             n_classes,
             activation=None)
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

print('y_pred : ', y_pred)
print('y_pred_cls', y_pred_cls)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def optimize(epoch):
  print('Epoch : ', epoch+1)
  for k in range(num_iterations):
    b_x, b_y_true = load_batch(k)
    feed_dict = {x:b_x, y_true:b_y_true}
    sess.run(optimizer, feed_dict=feed_dict)

    if k % acc_every == 0 and k!=0:
      y_t_cls, acc, y_p_cls = sess.run([y_true_cls, accuracy, y_pred_cls], feed_dict=feed_dict)
      print('Accuracy after {} iterations : {} \n y_pred_cls : {} \n y_true_cls : {}'.format(k+1, acc, y_p_cls, y_t_cls))

for e in range(epochs):
  optimize(e)



