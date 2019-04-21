''' Mnist For Deep Neural Network '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os

model_path = './model'
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

accuracy = 0
with tf.Session() as sess:
  saver = tf.train.import_meta_graph(model_path+'/mnist-dnn-model.ckpt.meta')
  saver.restore(sess, tf.train.latest_checkpoint(model_path))
    
  graph = tf.get_default_graph()

  inputs = graph.get_tensor_by_name("input:0")
  labels = graph.get_tensor_by_name("label:0")
  pred = graph.get_tensor_by_name("pred:0")

  while(True):
    x_data, y_data = mnist.test.next_batch(1)

    img = Image.fromarray((x_data*255).astype(np.uint8).reshape( ( 28, 28 ) ), 'P').resize((160, 160))
    img.show()

    ''' Inference image '''
    inference = sess.run(pred, feed_dict={ inputs: x_data })
    print('This picture is recognized as the number {}'.format(inference[0]))
    wait = input("Press any key to continue...")


