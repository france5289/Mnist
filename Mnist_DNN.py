import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

BATCH_SIZE = 64
sum = 0
steps = 2000

tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# 1. Construct a graph representing the model.
x = tf.placeholder(tf.float32, [BATCH_SIZE, 784], name="input") # Placeholder for input.
y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name="label") # Placeholder for labels.

W_1 = tf.Variable(tf.random_uniform([784, 100])) # 784x100 weight matrix.
b_1 = tf.Variable(tf.zeros([100])) # 100-element bias vector.
layer_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1) # Output of hidden layer.

W_2 = tf.Variable(tf.random_uniform([100, 10])) # 100x10 weight matrix.
b_2 = tf.Variable(tf.zeros([10])) # 10-element bias vector.
layer_2 = tf.matmul(layer_1, W_2) + b_2 # Output of linear layer.

# 2. Add nodes that represent the optimization algorithm.
with tf.name_scope('Loss'):
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=layer_2)
  tf.summary.scalar('loss',tf.reduce_mean(loss))
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(layer_2,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
filewriter = tf.summary.FileWriter('Tensorboard',tf.Session().graph)

# 3. Execute the graph on batches of input data.
with tf.Session() as sess: # Connect to the TF runtime.
  sess.run(tf.global_variables_initializer()) # Randomly initialize weights.
  for i in range(steps): # Train iteratively for NUM_STEPS.
    x_data, y_data = mnist.train.next_batch(BATCH_SIZE) # Load one batch of input data.
    train_accuracy = accuracy.eval(feed_dict={x: x_data, y: y_data})
    sum = sum + (train_accuracy * BATCH_SIZE)
    if i % 100 == 0:
      result = sess.run(merged, {x: x_data, y: y_data})
      filewriter.add_summary(result, i)
      print('step %d, training accuracy %g' % (i, train_accuracy))
    sess.run(train_op, {x: x_data, y: y_data}) # Perform one training step.
    
  print('test accuracy %g' % (sum/(BATCH_SIZE*steps)))
