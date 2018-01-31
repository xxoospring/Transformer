import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import sys
sys.path.append('../dataGenerate/')
from data_in import *


def get_data(batch_size=100):
    image, label = decode_from_tfrecords()
    batch_image, batch_label = get_batch(image, label, batch_size=batch_size)  # batch:10
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch_image_np, batch_label_np = sess.run([batch_image, batch_label])
        coord.request_stop()  # queue should be closed
        coord.join(threads)
    return batch_image_np, batch_label_np




net = input_data([100, 100, 1], name='input')
# net = conv_2d(net, )
net = conv_2d(net, 32, 5, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 5, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 500, activation='relu', regularizer='L2')
net = dropout(net, 0.5)
net = fully_connected(net, 2, activation='softmax', regularizer='L2')

net = regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.01)
model = tflearn.DNN(net, tensorboard_verbose=0)


image, label = decode_from_tfrecords(data_lst,)
batch_image_t, batch_label_t = get_batch(image, label, [100, 100, 1], batch_size=6000)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    batch_image, batch_label = sess.run([batch_image_t, batch_label_t])
    model.fit(batch_image, batch_label,n_epoch=10, validation_set=[batch_image, batch_label],show_metric=True,batch_size=50,)
    print('*************%s***************')
    coord.request_stop()  # queue should be closed
    coord.join(threads)
