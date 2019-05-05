import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

BATCH_SIZE = 128
sum = 0
steps = 2000

tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  '''
  "strides": is 1-D tensor of length 4, following NHWC format:(Num_samples x Height x Width x Channels) 
  '''
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Define the TensorFlow graph
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1]) # [-1,28,28,1] : batch size => determined during traning, 28*28 image, channel:1 => gray-scale image

#Convolution layer 1
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

W_conv1 = weight_variable([5, 5, 1, 32]) # filter size: 5*5*1 , number of filters:32
b_conv1 = bias_variable([32])

#Activation layer RELU
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#Activation layer MAXPOLLING
h_pool1 = max_pool_2x2(h_conv1)

#Convolution layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Flatten
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#NN layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#NN layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('entropy'):
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
  tf.summary.scalar('cross_entropy', cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
filewriter = tf.summary.FileWriter('Tensorboard',tf.Session().graph)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(steps):
    batch = mnist.train.next_batch(BATCH_SIZE)
    train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
    sum = sum + (train_accuracy * BATCH_SIZE)
    if i % 100 == 0:
      result = sess.run(merged, {x: batch[0], y_: batch[1], keep_prob: 1.0})
      filewriter.add_summary(result, i)
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % (sum/(BATCH_SIZE*steps)))
